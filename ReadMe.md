# 设置虚拟环境

python3 -m venv venv

激活虚拟环境 
. venv/bin/activate

然后安装相应的包 

pip install mindspore

安装完后用 pip list 查看

使用Pycharm 创建项目够需要设置项目的解释器

这里需要选择System Interpreter(系统解释器)才有效果（经实践），设置好后就能查到虚拟环境中安装的包


# 测试

运行 helloworld.py 得到
[ 4. 10. 18.]

运行dr.py 
得到准确率 {'accuracy': 0.9906850961538461}


运行verify_model.py(验证模型核心代码)
输出结果:
Predicted: "[9 6 6 4 4 9]", Actual: "[9 1 1 5 6 6]"

需要将某些包更新到较久的版本
Matplotlib 3.5.3
numpy 1.20
mindspore 1.7
mindvision 0.1.0