Transformer Machine Translation
==
基于Transformer架构的德语到英语机器翻译系统，在Multi30K数据集上训练实现。

📋项目简介
==
本项目实现了一个完整的Transformer机器翻译模型，支持德语到英语的翻译任务。包含数据预处理、模型训练、超参数调优和推理评估的全流程。

🏗️项目结构
==
<img width="426" height="465" alt="微信图片_20251031003210_167_65" src="https://github.com/user-attachments/assets/59f4d056-4189-4abd-9823-1deb6fb96170" />

⚙️环境配置与依赖
==
    #创建conda环境
    conda create --name transformer python=3.9
    conda activate transformer
    # 或使用venv创建虚拟环境
    python -m venv transformer-env
    source transformer-env/bin/activate  # Linux/Mac
    # 或
    transformer-env\Scripts\activate  # Windows
    
     # 安装PyTorch (根据CUDA版本选择)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    # 安装其他依赖
    pip install -r requirements.txt

🚀 实现步骤
==
数据预处理
---
<img width="399" height="80" alt="73170d521fb9cc6b865632bd07061d0c" src="https://github.com/user-attachments/assets/35ae108b-5d56-42b6-b9e7-9e6173d53a3b" />

训练模型
---
<img width="400" height="273" alt="30946e0e235db03693e117fe09e822fe" src="https://github.com/user-attachments/assets/f08ed8a9-5257-4b68-a4b9-316851170564" />

超参数敏感性分析复现
---
<img width="572" height="64" alt="9a1c16da5e2a12c37385a58b934004e4" src="https://github.com/user-attachments/assets/83d8b5ad-442a-4f51-8a65-67bb30f9f5d5" />

📈实验结果
==
<img width="3570" height="1166" alt="training_results" src="https://github.com/user-attachments/assets/5f13b5b9-a949-4039-9721-9092c6865f53" />
<img width="4470" height="1466" alt="ablation_study" src="https://github.com/user-attachments/assets/601e1d08-4da8-47b8-8724-8c19601c7bd5" />
<img width="2370" height="1765" alt="图片24" src="https://github.com/user-attachments/assets/849762ff-e204-4ac9-86b9-1705b2b17f72" />
<img width="2370" height="1765" alt="图片23" src="https://github.com/user-attachments/assets/179049f7-d0de-41cc-8475-360fd638fd02" />
<img width="2370" height="1765" alt="图片22" src="https://github.com/user-attachments/assets/6366b9b9-2096-4a7c-9c5e-53f8f4c43972" />
<img width="2370" height="1765" alt="图片21" src="https://github.com/user-attachments/assets/b87622f2-e944-4a6b-a5d9-7179e3f67410" />
<img width="2371" height="1765" alt="图片20" src="https://github.com/user-attachments/assets/b5906c52-2b50-4284-bf0f-a96854bec8d2" />
<img width="2370" height="1752" alt="图片19" src="https://github.com/user-attachments/assets/e9a0dc44-b75e-4b1f-b955-d7c8876ca530" />

🤝贡献
==
欢迎提交Issue和Pull Request来改进本项目。

📄联系方式
==
邮箱：xinniliang@163.com


