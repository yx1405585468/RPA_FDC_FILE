# RootCauseAnalysis Pyspark Project

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 简介

RootCauseAnalysis Pyspark Project 是一个使用 Pyspark 构建的大数据处理和分析工具。该项目的目标是为用户提供各类算法实现对芯片制造中出现的问题数据的分析。

## 项目结构

```plaintext
ikas-rca-job/
│  .gitignore
│  README.md
│  requirements.txt
│              
├─config
│      rca_config.properties
│      spark_config.yaml
│      __init__.py
│      
├─src
│  │  main.py
│  │  __init__.py
│  │  
│  ├─correlation
│  │  │  building_dataframe.py
│  │  │  by_zone_json_config.py
│  │  │  correlation_algorithm.py
│  │  │  correlation_main.py
│  │  │  parse_json_to_config.py
│  │  │  parse_json_to_sql.py
│  │  │  __init__.py
│  │  │  
│  │  ├─by_site_algorithms
│  │  │      compare_inline.py
│  │  │      compare_wat.py
│  │  │      main.py
│  │  │      __init__.py
│  │  │      
│  │  ├─by_wafer_algorithms
│  │  │      compare_inline.py
│  │  │      compare_process.py
│  │  │      compare_qtime.py
│  │  │      compare_uva.py
│  │  │      compare_wat.py
│  │  │      main.py
│  │  │      __init__.py
│  │  │      
│  │  ├─by_zone_algorithms
│  │  │      compare_inline.py
│  │  │      compare_wat.py
│  │  │      main.py
│  │  │      __init__.py
│  │  │      
│  │  ├─common_process
│  │  │      corr_base_alogrithm.py
│  │  │      data_preprocessing.py
│  │  │      __init__.py
│  │
│  ├─defect
│  │  │  build_query.py
│  │  │  defect_algorithm.py
│  │  │  defect_main.py
│  │  │  __init__.py
│  │
│  ├─exceptions
│  │  │  rca_base_exception.py
│  │  │  __init__.py
│  │          
│  ├─fdc_advanced
│  │      fdc_advanced_algorithm.py
│  │      fdc_advanced_main.py
│  │      __init__.py
│      
│  ├─inline
│  │      build_query.py
│  │      inline_bysite_algorithm.py
│  │      inline_bywafer_algorithm.py
│  │      inline_byzone_algorithm.py
│  │      inline_main.py
│  │      __init__.py
│      
│  ├─utils
│  │  │  app_config.properties
│  │  │  config_utils.py
│  │  │  http_utils.py
│  │  │  read_jdbc_executor.py
│  │  │  __init__.py
│  │          
│  ├─uva
│  │      build_query.py
│  │      uva_algorithm.py
│  │      uva_main.py
│  │      __init__.py
│      
│  ├─wat
│  │      build_query.py
│  │      wat_bysite_algorithm.py
│  │      wat_bywafer_algorithm.py
│  │      wat_byzone_algorithm.py
│  │      wat_main.py
│  │      __init__.py
```

### 文件说明

- **config/**: 存放项目配置文件，包括 Spark 配置和 RCA（Root Cause Analysis）相关的属性配置文件。
- **src/**: 项目源代码目录。
  - **correlation/**: 处理相关性分析的模块和算法。
  - **defect/**: 与缺陷检测和分析相关的代码。
  - **exceptions/**: 项目中自定义的异常处理模块。
  - **fdc_advanced/**: 用于FDC的算法和实现。
  - **inline/**: 包含inline相关的算法和数据处理。
  - **utils/**: 实用工具模块，提供各种通用功能，如配置读取、HTTP 请求处理等。
  - **uva/**: 包含UVA算法和实现。
  - **wat/**: 包含WAT算法和实现。

## 依赖

在开始运行项目之前，需要安装以下依赖：

- **Python 版本**: 3.9 及以上
- Apache Spark 3.3.4
- 其他依赖可以通过运行以下命令安装：

```bash
pip install -r requirements.txt
```

## 配置

项目配置文件存放在 `config/` 目录下。

- `rca_config.properties`: 包含 RCA 相关的属性配置。
- `spark_config.yaml`: Spark 相关的配置。

确保根据你的环境正确设置这些配置文件，特别是 Spark 的配置。

## 运行

要运行该项目，你可以使用以下命令：

```bash
python src/main.py
```

该命令将启动主程序，并根据配置文件执行相关的算法和数据处理任务。

## 示例用途

你可以通过修改 `config/rca_config.properties` 和 `config/spark_config.yaml` 文件来调整项目的行为。项目支持多种分析模式，可以根据不同的需求进行定制。

例如：

- 对生产数据进行根因分析，找出潜在的生产问题。
- 分析不同区域的数据差异，识别异常区域。
- 通过相关性分析找出关键影响因素。

## 贡献

## 许可证
