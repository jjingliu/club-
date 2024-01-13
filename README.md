# Journal Club 学期汇总

## 空间转录组相关

### 数据整合/对齐

**1**  [`Partial alignment of multislice spatially resolved transcriptomics data`](https://www.biorxiv.org/content/10.1101/2023.01.08.523162v1.full.pdf)

**期刊/时间** `Genome reserch / 2023 10` 

**关键词** `Alignment of spatial transcriptomics`  `PASTE2`

**问题描述** 对齐空间转录组数据的方法，是之前的PASTE的进一步工作，考虑了空间转录组切片只有部分对齐的情况

PASTE在进行切片对齐时假设两个切片可以在整体的2D切片内（所有点）进行对齐。即两个切片整体在生理上、技术上都具有高度的相似性。然而这通常并不是一个合理的假设，因为组织解剖和阵列（捕获区）放置在一致性上存在技术困难，并且相邻切片之间组织形态可能也难免存在差异。有时可能相邻切片之间沿z轴（垂直于切片的轴）只有部分组织重叠:
<img src="fig/paste2_1.png">
<img src="fig/paste2_2.png">
s被认为是表示两切片之间重叠占比的参数，当g_i=1/n时，将有占比大约为s的细胞参与了对齐。

**2**  [`STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping`](https://www.nature.com/articles/s41467-023-43915-7)

**期刊/时间** `Nature Communications / 2023 12` 

**关键词** `Alignment of spatial transcriptomics`  `LDDMM`

**问题描述** 使用一个现有的图像映射算法LDDMM来对齐空间转录组数据
<img src="fig/STalign.png">


**3**  [SPIRAL: integrating and aligning spatially resolved transcriptomics data across different experiments, conditions, and technologies](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03078-6)

**期刊/时间** `Nature Computational Science  / 2023 10` 

**关键词** `integrating and aligning of spatial transcriptomics` `SPIRAL`

**问题描述** SPIRAL 由两个连续模块组成： SPIRAL-integration（基于图域适应的数据集成）和 SPIRAL-alignment（基于集群感知的优化传输协调对齐）
<img src="fig/SPIRAL.png">


**4**  [Integrating spatial transcriptomics data across different conditions, technologies and developmental stages](https://www.biorxiv.org/content/10.1101/2022.12.26.521888v1.full.pdf)

**期刊/时间** `Nature Computational Science  / 2023 10` 

**关键词** `Integrating spatial transcriptomics data` `STAligner`

**问题描述** STAligner 的图注意神经网络，用于整合和对齐 ST 数据集，实现空间感知数据整合、同步空间域识别和下游比较分析。
<img src="fig/STAligner.png">


**5**  [`Latent feature extraction with a prior-based self-attention framework for spatial transcriptomics`](https://genome.cshlp.org/content/33/10/1757.full)

**期刊/时间** `Genome reserch / 2023 10` 

**关键词** `Latent feature extraction` `self- attention`  `PAST`

**问题描述** 基于自注意力框架来提取空间转录组数据的潜在特征
<img src="fig/past.png">


**6**  [`MUSTANG: multi-sample spatial transcriptomics data analysis with cross-sample transcriptional similarity guidance`](https://www.biorxiv.org/content/10.1101/2023.09.08.556895v1.full.pdf)

**期刊/时间** `bioRxiv /2023.09` 

**关键词** `transcriptomics data analysis` `grouped horseshoe distribution` `spot similarity graph` `MUSTANG`

**问题描述** 空间转录组数据整合分析，最终得到的是所有spot的新的表示，并不强调空间坐标对应
<img src="fig/MUSTANG.png">


###空间域/空间可变基因识别
**1**  [`STAMarker: determining spatial domain-specific variable genes with saliency maps in deep learning`](https://www.biorxiv.org/content/10.1101/2022.11.07.515535v1.full.pdf)

**期刊/时间** `Nucleic Acids Research / 2023.10` 

**关键词** `graph-attention autoencoders` `multilayer perceptron classifiers` `saliency map computation`

**问题描述** 识别特定于空间域的空间可变基因
<img src="fig/STAMarker.png">

**2**  [`Probabilistic cell / domain-type assignment of spatial transcriptomics data with SpatialAnno`](https://www.biorxiv.org/content/10.1101/2023.02.08.527590v1.full.pdf)

**期刊/时间** `Nucleic Acids Research / 2023.11` 

**关键词** `factor model` `Potts model`

**问题描述** 利用大量非标记基因以及有关标记基因的“定性”信息进行空间转录组数据集注释
<img src="fig/SpatialAnno.png">

**3**  [`Mapping the topography of spatial gene expression with interpretable deep learning`](https://www.biorxiv.org/content/10.1101/2023.10.10.561757v1.full.pdf)

**期刊/时间** `bioRxiv /2023.10` 

**关键词** `Isodepth` `spatial transcriptomics data` `spatial domain` `GASTON`

**问题描述** 从 SRT 数据中识别具有空间变化表达的空间域和/或基因
<img src="fig/GASTON.png">



