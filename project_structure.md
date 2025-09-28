# LLM-SNGAT 项目文件结构

## 完整的项目目录结构

```
llm-sngat/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── setup.py                    # 项目安装配置
├── config.py                   # 配置文件
├── utils.py                    # 工具函数
├── llm_sngat.py               # 主要实现代码
├── run_experiment.py          # 实验运行脚本
├── demo.ipynb                 # Jupyter演示笔记本
├── .gitignore                 # Git忽略文件
├── LICENSE                    # 开源许可证
│
├── data/                      # 数据目录
│   ├── aqua_dataset.json     # AQua数据集（可选）
│   └── sample_problems.json  # 示例问题
│
├── results/                   # 结果输出目录
│   ├── llm_sngat_results_*.xlsx
│   ├── raw_results_*.json
│   └── summary_report_*.txt
│
├── figures/                   # 图表输出目录
│   ├── model_comparison_*.png
│   └── detailed_analysis_*.png
│
├── logs/                      # 日志目录
│   └── llm_sngat_*.log
│
├── tests/                     # 测试代码
│   ├── __init__.py
│   ├── test_dataset_loader.py
│   ├── test_simulator.py
│   ├── test_processor.py
│   └── test_analyzer.py
│
├── docs/                      # 文档目录
│   ├── source/
│   ├── build/
│   └── methodology.md
│
└── examples/                  # 示例代码
    ├── basic_usage.py
    ├── advanced_analysis.py
    └── custom_models.py
```

## 主要文件说明

### 核心实现文件

1. **llm_sngat.py** - 主要实现代码
   - `AQuaDatasetLoader`: 数据集加载器
   - `StudentSimulator`: 学生模拟器
   - `LLMResponseSimulator`: LLM响应模拟器
   - `LLMSNGATProcessor`: 主处理器
   - `ResultAnalyzer`: 结果分析器

2. **config.py** - 配置管理
   - 模型配置（GPT-4o, O1-preview, DeepSeek-R1等）
   - 仿真参数配置
   - 绘图样式配置
   - 路径配置

3. **utils.py** - 工具函数
   - 日志设置
   - 数据保存/加载
   - 统计计算
   - 问题生成器

4. **run_experiment.py** - 实验运行器
   - 命令行接口
   - 批量实验运行
   - 结果汇总和报告生成

### 使用文件

5. **demo.ipynb** - Jupyter演示笔记本
   - 完整的演示流程
   - 交互式分析
   - 可视化结果

6. **requirements.txt** - 依赖包列表
   ```
   numpy>=1.21.0
   pandas>=1.3.0
   scipy>=1.7.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   faker>=8.0.0
   openai>=0.27.0
   scikit-learn>=1.0.0
   openpyxl>=3.0.0
   SciencePlots>=2.0.0
   jupyter>=1.0.0
   ```

### 配置和文档

7. **setup.py** - 包安装配置
8. **.gitignore** - Git版本控制忽略文件
9. **LICENSE** - 开源许可证

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/yourusername/llm-sngat.git
cd llm-sngat

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或安装为包
pip install -e .
```

### 2. 基本使用

```bash
# 运行演示
python llm_sngat.py

# 运行完整实验
python run_experiment.py --models GPT-4o O1-preview DeepSeek-R1

# 使用Jupyter笔记本
jupyter notebook demo.ipynb
```

### 3. 高级使用

```bash
# 自定义配置运行
python run_experiment.py \
  --models GPT-4o DeepSeek-R1 \
  --n-students 200 \
  --form-size 60 \
  --common-sizes 5 10 15 20 25 \
  --replications 3 \
  --use-real-llm

# 指定输出目录
python run_experiment.py --output-dir ./my_results/
```

## 项目特点

### 🚀 功能特点
- **多模型支持**: GPT-4o, O1-preview, DeepSeek-R1等
- **灵活配置**: 可自定义学生数量、测试规模、锚点项目数量
- **双重等值方法**: Tucker线性等值和Levine观察分数等值
- **可视化分析**: 完整的图表和统计分析
- **标准误差计算**: 基于Delta方法的精度评估

### 📊 输出结果
- **Excel报告**: 详细的数值结果和统计分析
- **可视化图表**: 标准误差趋势、模型比较、分布分析
- **JSON数据**: 原始结果用于进一步分析
- **文本报告**: 自动生成的总结和建议

### 🔧 扩展性
- **模块化设计**: 易于添加新的LLM模型
- **插件架构**: 支持自定义等值方法
- **API支持**: 可集成到其他教育测评系统
- **数据格式**: 支持多种数据集格式

## API密钥配置

在使用真实LLM时，需要设置相应的API密钥：

```bash
# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 或在.env文件中设置
echo "OPENAI_API_KEY=your-key" > .env
```

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了此项目，请引用：

```bibtex
@software{llm_sngat,
  title={LLM-SNGAT: LLM-Simulated Nonequivalent Groups with Anchor Test},
  author={Research Team},
  year={2025},
  url={https://github.com/yourusername/llm-sngat}
}
```
