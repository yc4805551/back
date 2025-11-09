import ollama
import sys

print(f"--- Running Ollama Test Script ---")
print(f"Python executable: {sys.executable}")
print(f"Ollama library path: {ollama.__file__}")
print("-" * 30)

try:
    # 初始化 Ollama 客户端
    print("尝试连接到 Ollama 服务...")
    client = ollama.Client(host="http://127.0.0.1:11434")
    print("✓ Ollama 客户端初始化成功")
    
    # 获取可用模型列表
    print("\n获取可用模型列表...")
    response = client.list()
    print(f"Ollama 响应原始数据: {response}")
    
    # 解析模型列表
    models = response.get('models', [])
    print(f"\n找到 {len(models)} 个可用模型:")
    
    # 遍历并打印每个模型的详细信息
    for i, model in enumerate(models, 1):
        print(f"\n模型 {i}:")
        # 检查 model 是对象还是字典
        if hasattr(model, '__dict__'):
            # 是对象，访问其属性
            print(f"  类型: 对象")
            print(f"  model: {getattr(model, 'model', 'N/A')}")
            print(f"  modified_at: {getattr(model, 'modified_at', 'N/A')}")
            print(f"  size: {getattr(model, 'size', 'N/A')}")
        elif isinstance(model, dict):
            # 是字典，访问其键值
            print(f"  类型: 字典")
            for key, value in model.items():
                print(f"  {key}: {value}")
        else:
            print(f"  类型: 未知 ({type(model)})")
            print(f"  内容: {model}")
    
    # 测试特定功能
    print("\n--- 测试完成 ---")
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    print(f"错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
