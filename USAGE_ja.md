# 使い方マニュアル（日本語）

論文「反応性サンプリングとデータキュレーションによる粒界機械学習力場の構築と、
Li₆PS₅X 大規模粒界輸送への応用」で使ったスクリプトの手順書です。コマンドは論文と同じ設定で書いてあります。
全体像とスクリプト・論文結果の対応表は [`README.md`](README.md)、英語版の手順書は [`USAGE_en.md`](USAGE_en.md) です（内容は同じ）。

共通の約束:

- `--model` には `pretrained`（MatterSim v1.0.0-1M そのまま。論文では universal / generic と呼んでいます）か、
  微調整済みの `.pth`（`v5D.pth`）のパスを渡します。
- 粒界（GB）の法線は **x** です。GB モデル（Lx = 69.434 Å）の2枚の GB 面は 0.25·Lx = 17.36 Å と 0.75·Lx = 52.07 Å にあります。
  GB 窓は各面の ±3.5 Å、バルク窓は x = 30–40 Å です。
- `python <スクリプト> -h` で全オプションが出ます。各スクリプトの冒頭コメントに、何をするか・なぜそうするかを書いてあります。
- バッチジョブ用のラッパー（pjsub スクリプト等）は入れていません。ここに書いたコマンドが各ジョブで実際に流した中身です。

---

## 0. 環境

```bash
pip install -r requirements.txt     # mattersim==1.2.0（固定）, torch, ase, numpy, scipy, scikit-learn
```

- **mattersim は 1.2.0 固定です。** 1.2.5 では埋め込み空間が変わり、凍結したクラスタと選別が再現しません。
- ASE ≥ 3.24 では `full_3x3_to_voigt_6_stress` が `ase.stress` に移りました。MatterSim を読む各スクリプトの先頭に互換 shim が入っているので、何もしなくて大丈夫です。
- DFT は Quantum ESPRESSO 6.7 です。設定は PBE+D3、ウルトラソフト擬ポテンシャル、ecutwfc 48.9 Ry / ecutrho 440.5 Ry、
  conv_thr 1e-6、k 点は ⌈k_len/|aᵢ|⌉（k_len = 26）。どれも `training/make_scf_inputs.py` の既定値になっています。
- エネルギーは凝集エネルギーで保存しています（Ry→eV = 13.6058、bohr→Å = 0.529177）。元素ごとのゲージ
  （`training/energy_gauge_v4l0.json`）で MatterSim のゼロ点に合わせます。

---

## 1. ODPMD による反応経路サンプリング（`sampling/`）

```bash
python sampling/prepare_runs.py --src /path/to/equilibrated_inputs --out runs \
    --vatom 15 17 25 --temp 300 600 900
bash runs/stage_model.sh /path/to/matersim_model.pth     # 全ランにモデルをコピー
cd runs/v17.0_T600 && python /path/to/sampling/ODPMD.py in.test > output.txt
```

- 論文では 206 原子の GB セル Li₉₄P₁₆S₇₈Br₉Cl₉ で、V/N ∈ {15, 17, 25} Å³/atom × T ∈ {300, 600, 900} K の 12 ランを流しました（36,490 フレーム）。
- `PS4.log` に P–S 結合の切断が記録されます。骨格が壊れている間はフレームを密に書き出します。
- 解離と新規性の診断は `sampling/ps4_probe.py` です。

---

## 2. キュレーション（`curation/`）

### 2.1 物理整合性フィルタ

```bash
python curation/filter_frames.py listdata.txt --system system.json \
    --coord_rule P:S:4 --coord_rcut 2.8 --out frame_filter.npy
```

`listdata.txt` には DeePMD のシステムディレクトリを1行に1つ書きます。読むのは `set.000` だけです。
ディレクトリを自分で glob して足すと結果が合わなくなります。

### 2.2 PS₄ 健全性の監査

```bash
python curation/audit_ps4.py v3_train_reduce_emb.extxyz --out audit_ps4/frames.csv
```

各フレームについて、P の配位、PS₄ 頂点にあるハロゲンの数（`n_vertexX`）、遊離 S を記録します。
`n_vertexX > 0` のフレームは古い構造生成ルールが作ったアーティファクトなので、4.3 で除きます（5271 → 4074 フレーム）。

---

## 3. 記述子空間での選別（`selection/`）

```bash
python selection/extract_embeddings.py listdata.txt --ckpt matersim_model.pth
python selection/cluster_embeddings.py --emb embeddings.npy --meta emb_meta.npy
python selection/select_from_clusters.py listdata.txt --nsel 4000 --alloc prop
```

後から来るデータは、**凍結した** `cluster_model.npz` を基準に判定します:

```bash
python selection/screen_candidates.py -h     # DFT を回す前に QE 候補を順位付け
python selection/assign_clusters.py -h       # 新フレームをクラスタに割り当てるか、novel と判定
python selection/update_embeddings.py -h     # 累積の記述子ストアに追記
```

`emb_cache.py` はこれらと同じディレクトリに置いてください。

---

## 4. DFT データと v5D 力場（`training/`）

### 4.1 DFT 単点計算

```bash
python training/make_scf_inputs.py --traj frames.extxyz --out dft_frames --klen 26 --stress \
    --pseudo_dir /path/to/pseudo
# dft_frames/*.in を pw.x で流してから
python training/scf_to_deepmd.py -h        # QE → DeePMD（virial を入れるときは --with_stress）
```

Subset A/B のフレームは応力付きです。移動経路フレームと AL フレームは応力なしなので、`--stress` を付けません。

### 4.2 移動経路データ（Subset C）用の健全な構造

Li 欠損バルク（遊離 S²⁻ をハロゲンに置換し、PS₄ には触らない）。`v3_train_reduce_emb.extxyz` があるディレクトリで実行します:

```bash
mkdir -p audit_ps4/bulkdef
python training/build_lidef_bulk.py          # audit_ps4/bulkdef/{stoich,lidef}_*.extxyz を出力
```

DFT サイズの GB セル（260 原子、Σ5(210)、PS₄ は健全）。次のコマンドで論文の構造がそのまま再現できます:

```bash
X=l6ps5cl_210.xyz
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 1 --seed 0 --out gbS_rnd0.extxyz   # ランダム
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 1 --seed 1 --out gbS_rnd1.extxyz
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 4 --lam 3 --seed 2 --out gbS_seg2.extxyz   # GB 寄り
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 4 --lam 3 --seed 3 --out gbS_seg3.extxyz
# Li2S 欠損 GB（GB 面ごとに Li2S を 2 / 4 単位除去。NEB でホップさせる Li 0,50,43,63 は保護）
python application/gb_li2s_defect.py --data gbS_seg2.extxyz --n_per_gb 2 --protect 0,50,43,63 --out gbS_d2.extxyz
python application/gb_li2s_defect.py --data gbS_seg2.extxyz --n_per_gb 4 --protect 0,50,43,63 --out gbS_d4.extxyz
```

DFT に回す反応性 MD のフレームと NEB のイメージ:

```bash
python training/md_harvest.py --data gbS_rnd0.extxyz --model pretrained --temp 900 \
    --nsteps 10000 --equil 2000 --dump_every 500 --out mdH_rnd0_pre_900K_s0     # 600 K と 900 K で実施
python validation/neb_barrier.py --data gbS_rnd0.extxyz --list_hops --gb-axis x --gb-pos 0.25,0.75
python validation/neb_barrier.py --data gbS_rnd0.extxyz --model v5D.pth --hop 0 50 \
    --images 7 --climb --k 0.1 --relax_fmax 0.05 --comp_free_s --out nebS_rnd0_0_50
python training/make_scf_inputs.py --traj nebS_rnd0_0_50_path.extxyz --out dft_nebS_rnd0_0_50 \
    --klen 26 --min_dist 0 --fmax_max 1e9
```

**GB セルでは必ず `--comp_free_s` を付けてください。** 付けないと、空孔の電荷補償に PS₄ 頂点の S が選ばれることがあります（旧挙動）。

train / test の分割は構造単位で行い、フレーム単位では分けません。GB の test は rnd1・seg3・d2、
バルクの test は Br6Cl6Li44P8S36 です。各組成系列の混合組成も test 側に残します。

### 4.3 クリーンな学習セットの組み立て

```bash
python training/assemble_v5_train.py \
    --base v5build/v3_train_reduce_emb.extxyz --frames_csv audit_ps4/frames.csv \
    --gb_root ps4ok --bulk_root bulkdef --neb_mult 3 --md_mult 1 \
    --outA v5build/v5_trainA.extxyz --outB v5build/v5_trainB.extxyz \
    --out_dft v5build/v5_clean_dft.extxyz
```

論文で使ったのは `v5_trainB` です（反応性ベースから `n_vertexX>0` を除き、クリーンな DFT を足したもの）。
`v5_trainA` はベースを全部残した対照です。

### 4.4 GB 芯の能動学習

前世代の微調整モデルは、700 K で GB 芯の PS₄ を分解させていました。AL ループで、そうした構造の DFT エネルギーをモデルに教えます:

```bash
# 1) GB 芯 1 枚のまわりを DFT サイズで切り出す（y/z 周期、x に真空、外側 2.5 Å 固定）
python training/cut_gb_core.py --data gbV2_uni.extxyz --gb-x 17.36 --half_width 7 --vacuum 15 --out cut_uni_gb1.extxyz
# 2) 候補モデルごとに拘束緩和 + 700 K MD（50 step ごとに PS4 健全性を記録）
python training/relax_md_cut.py --data cut_uni_gb1.extxyz --model v5B.pth --temp 700 --steps 10000 --out rx_uni_gb1_v5B
# 3) 壊すモデルからは分解イベントのフレーム、健全なモデルからは等間隔のフレームを選ぶ（分割はカット単位）
python training/select_al_frames.py --dir . --train_cuts n8_gb1,n8_gb2,uni_gb1 --test_cuts uni_gb2 --out al1
# 4) DFT 単点（応力なし）。カットの拘束 DFT 緩和もする場合は 2 行目
python training/make_scf_inputs.py --traj al1_train.extxyz --out al1train --klen 26 --pseudo_dir /path/to/pseudo
python training/make_relax_cut_pwin.py --data cut_n8_gb1.extxyz --klen 26 --pseudo_dir /path/to/pseudo --out dftrx_n8_gb1
# 5) 収束した QE 出力を、同じ凝集エネルギーゼロ点の extxyz に変換
AL_DIR=/path/to/al python training/al_to_extxyz.py      # → v5C_al_train.extxyz / v5C_al_test.extxyz
```

### 4.5 v5D の微調整

```bash
MODEL=models/v4_l0.pth TRAINB=v5build/v5_trainB.extxyz \
AL_TRAIN=al/v5C_al_train.extxyz AL_TEST=al/v5C_al_test.extxyz VALID=v3_test_emb.extxyz \
sh training/run_v5D.sh
```

スクリプトがすること:

1. **エネルギーゲージ。** `energy_gauge_v4l0.json` があればそれを使い、なければ再 fit します。
   ゲージは、微調整の**出発点**のチェックポイント（`v4_l0`）に対して fit しないといけません。
   残差が Huber δ = 0.2 eV/atom を超えるデータセットが残っていれば、学習せずに止まります。
   別のチェックポイントで fit したゲージだと、Li₂S 梯子の AL フレームで 1.4–4.5 eV/atom ずれて、そのフレームが学習されなくなります。
2. **組み立て。** `v5D_train.extxyz` = `v5_trainB` + AL × 6 を作ります。
3. **学習。** v4（λ = 0）から温間開始して 80 epoch。lr 1e-4、batch 16、Huber δ 0.2、力:エネルギー = 4、
   Li:P:S:Br:Cl = 3:1:1:1:1、応力なし、early stop なし、5 epoch ごとに保存。モデルは `results_v5D/best_model.pth` です。

チェックポイントのゲートでは、分解した PS₄ と健全な PS₄ のエネルギー差を `dftsp_relax/in/` の DFT 単点と比べ、
あわせて held-out の AL フレームでの E/F も見ます。`dftsp_relax/` と `al/` があるディレクトリで実行します:

```bash
python training/eval_v5C_gate.py --models results_v5D/best_model.pth,v5B.pth,pretrained --device cpu
```

L2-SP のスイープ（SI S5）は `finetune_mattersim.py --l2sp_lambda <λ>` で行いました。置き換え前の phaseB のレシピは `legacy/` にあります。

---

## 5. DFT との比較検証（`validation/`）

```bash
# held-out の Li ホップ障壁を DFT と同じ形状で点ごとに評価（論文では物理的な 25 経路）
python validation/eval_v5_barriers.py --model v5D.pth --device cuda \
    --gb_root ps4ok --bulk_root bulkdef --out eval_v5/bar_v5D.csv
python validation/eval_v5_barriers.py --model pretrained --device cuda \
    --gb_root ps4ok --bulk_root bulkdef --out eval_v5/bar_pretrained.csv
# 1 経路のイメージで MLIP と DFT を比較
python validation/neb_dft_compare.py --barrier nebS_rnd0_0_50_barrier.csv --qe_glob 'dft_nebS_rnd0_0_50/*.out'
# 同一形状での E / F / 応力のパリティ
python validation/parity.py --qe_glob 'exttest/*.out' --models pretrained v5D.pth
python validation/train_parity.py -h
# 弾性定数: ±0.005 の歪み 13 フレーム、イオン緩和 0.02 eV/Å。全モデルと DFT で同じフレームを使う
python validation/elastic_dft.py -h ; python validation/elastic_from_stress.py -h
```

論文では、DFT 障壁が 2 eV を超える 3 経路（端点が未収束）を障壁 MAE から除いています。

---

## 6. GB モデルと輸送（`application/`）

### 6.1 GB モデル

```bash
# Σ5(210) テンプレート（1448 原子）の一様装飾と偏析装飾
python application/gb_build_uniform.py --xyz test.GB.xyz --repeat 1,1,1 --enrich 2 --lam 3 --seed 0 --out gbV2_uni.extxyz
python application/gb_build_uniform.py --xyz test.GB.xyz --repeat 1,1,1 --enrich 4 --lam 3 --seed 0 --out gbV2_seg.extxyz
# アモルファス GB: GB スラブ（±0.10 Lx）だけを融解し、バルクは固定。v5D、dt 1 fs、10 + 10 + 5 ps
python application/amorphize_gb.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --gb-pos 0.25,0.75 --gb-axis x --half-width 0.10 --t_hot 1300 --t_cold 300 --dt 1.0 \
    --hot_ps 10 --quench_ps 10 --relax_ps 5 --seed 0 --out gbV2_amo.extxyz
# PS4 監査（P の配位 = 4、頂点ハロゲンなし、S–S / Li–Li / P–P の短距離対なし）
python application/audit_gb_ps4.py gbV2_uni.extxyz gbV2_seg.extxyz gbV2_amo.extxyz
```

- 上の 2 本の `gb_build_uniform.py` で、論文の `gbV2_uni` / `gbV2_seg` が全サイトの元素まで一致して再現します。
- `--repeat 1,1,1` を使ってください。y を半分にしたテンプレート（`make_mid`）は周期的になりません。
- T_hot = 2000 K ではスラブ内の PS₄ が壊れました。論文で使った設定は 1300 K です。作ったら必ず監査してください。

### 6.2 Li 拡散（4344 原子セル = gbV2 × (1,1,3)）

```bash
for M in v5D.pth pretrained; do for S in 0 1 2; do
  python application/diffusion_md.py --data gbV2_uni.extxyz --model $M --device cpu \
      --species Li --temp 700 --dt 2.0 --nsteps 50000 --equil 5000 --nsample 10 \
      --seed $S --repeat "1 1 3" --traj_every 50 --ckpt_every 2000 --resume \
      --regions "GB1:x:13.86:20.86 BULK:x:30:40 GB2:x:48.58:55.58" \
      --out msdV2uni_$(basename $M .pth)_700K_s$S
done; done
```

- 構造は `uni`・`seg`・`amo`、温度は 500・600・700・800 K で流します。
- 500 K は `--nsteps 100000`（200 ps）です。
- 50k step の 1 本は 32 CPU スレッドで約 15 時間（1 step ≈ 1.1 s）かかります。`--resume` でチェックポイントから続くので、何回かのジョブに分けて流せます。
- 流し終わったら PS₄ が壊れていないか確認してください: `python application/check_md_struct.py <out>_ckpt.extxyz`

領域ごとの拡散係数（seed ごと）と、seed 平均の Arrhenius 再 fit:

```bash
mkdir -p reg
for f in msdV2*_traj.npz; do b=${f%_traj.npz}
  python application/msd_regions.py $f --axis x \
      --regions "GB1:13.86:20.86 BULK:30:40 GB2:48.58:55.58" --lag 20 --out reg/reg_V2${b#msdV2}
done
python application/refit_arrhenius.py --reg_dir reg --out refit_summary.csv
```

`msd_regions.py` は、時間原点ごとに各 Li の領域を割り当て直し、短いラグだけで fit します。
700–800 K では Li は 29–48 ps で幅 7 Å の GB 窓を出てしまうので、初期位置だけで領域を決めると間違った値になります。
`refit_arrhenius.py` は温度ごとの D_GB/D_bulk（seed 平均 ± 標準誤差）と、seed ブートストラップの誤差付きの ΔEa(GB − bulk) を出します。

### 6.3 機構の指標

```bash
python application/mechanism_tier1.py --dir . --struct uni --temps 500 700 --seeds 0 1 2 --lag 20
python application/coord_tier2.py v5D=nemd_v5D_gbV2_uni_700K_z_F000_s0_ckpt.extxyz \
    pretrained=nemd_pretrained_gbV2_uni_700K_z_F000_s0_ckpt.extxyz --out coord_tier2.json
```

- Tier 1 は GB とバルクのそれぞれで、1 ps のケージ MSD、α₂ のピーク、20 ps での変位の中央値と 2 Å を超えて動いた Li の割合を出します。
- Tier 2 は、平衡化した 700 K の終状態で、Li の静的な配位数（r < 3.2 Å）と第一殻のハロゲン比率を出します。

### 6.4 骨格のダイナミクス（全原子、1448 原子セル）

```bash
python application/framework_md.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --temp 700 --dt 2.0 --nsteps 20000 --equil 5000 --dump_every 25 --seed 0 --out fwV2_v5D_uni_700K
python application/tier3_fwV2.py fwV2_*_fwtraj.npz --out tier3_fwV2_summary.json
```

- 論文で流したのは、`uni` の 500 K と 700 K、`seg` の 700 K で、どれも両方の力場です。
- 出力は、GB とバルクそれぞれの PS₄ の C₁(10 ps)、ハロゲンの MSD、Li の MSD です。

### 6.5 カラーチャージ NEMD

```bash
python application/nemd_ccd.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --temp 700 --dt 2.0 --nsteps 50000 --equil 10000 \
    --field 0.02 --field-axis z --profile-axis x --nbins 40 --block 250 \
    --seed 0 --repeat "1 1 3" --ckpt_every 2500 --resume --out nemd_v5D_gbV2_uni_700K_z_F020_s0
python application/nemd_analysis.py profile  "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz" --gb 17.36 52.08 --bulk 30 40 --out prof_v5D_gbV2_700K
python application/nemd_analysis.py haven    "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz"
python application/nemd_analysis.py arrhenius "nemd_v5D_gbV2_uni_*K_z_F*_s*_nemd.npz" --out arr_v5D_gbV2
python application/nemd_analysis.py rgb --perp "nemd_v5D_gbV2_uni_700K_x_F*_s*_nemd.npz" \
    --par "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz" --bulk 30 40 --ngb 2 --out rgb_v5D_gbV2_700K
```

- 線形応答は、700 K で F_e = 0, 0.005, 0.01, ±0.02, 0.04 eV/Å を流して確認しました。
- 本番は F_e = 0.02 eV/Å で、GB 面に平行（`z`）と垂直（`x`）の両方向、500–800 K、3 seed です。
- F_e = 0 のラン（`..._F000_..._ckpt.extxyz`）の終状態は、6.3 の平衡構造としても使っています。

### 6.6 空間電荷

```bash
python application/charge_profile_tavg.py --static gbV2_uni.extxyz     # 作ったままの構造: q_GB = -24 e、φ の段差 2.34 V
python application/charge_profile_tavg.py --dir . --tag 700K --out charge_tavg.npz
```

6.4 の 40 ps 全原子ランで ρ_q(x) を時間平均します（形式電荷、46 ビン、2.5 Å のガウス平滑化、ε_r = 15）。
期待される Δφ は、`uni` が 0.25 V（v5D）/ 0.21 V（generic）、`seg` が 0.51 / 1.18 V です。

### 6.7 ハロゲン配置の安定性（SI S6）

```bash
for C in uni seg; do
  python application/gb_segregation.py --data gbV2_$C.extxyz --model v5D.pth   --device cpu --nswap 20 --out segV2_v5D_$C
  python application/gb_segregation.py --data gbV2_$C.extxyz --model pretrained --device cpu --nswap 20 --out segV2_generic_$C
done
```

---

## 7. `legacy/`

phaseB 時代のパイプライン（`gbBIG_dec` セル、phaseB のレシピとゲージ、古い解析ドライバ）のスクリプトです。
経緯を残すために置いているだけで、論文の結果はどれもこれらに依存していません。詳しくは `legacy/README.md` を見てください。
