"""
测试 Embedding 检索功能
用于验证情景匹配检索是否正常工作
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from app import SENTENCE_TRANSFORMERS_AVAILABLE, embedding_models, proxy


def test_models_loading():
    """测试模型是否能正常加载"""
    print("=" * 60)
    print("测试 1: 检查 sentence-transformers 是否安装")
    print("=" * 60)

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers 未安装")
        print("请运行: pip install sentence-transformers torch")
        return False

    print("✅ sentence-transformers 已安装")

    try:
        print("\n正在加载情景匹配模型 (BAAI/bge-large-zh)...")
        retrieval_model = embedding_models.get_retrieval_model()
        print("✅ 模型加载成功")

        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False


def test_retrieval():
    """测试检索功能"""
    print("\n" + "=" * 60)
    print("测试 2: 测试情景匹配检索功能")
    print("=" * 60)

    # 使用测试用户
    test_user_id = "tukainan"  # 根据实际情况修改
    test_context = "深度学习模型结构"

    print(f"\n测试输入: {test_context}")
    print(f"测试用户: {test_user_id}")

    try:
        # 先检查该用户是否有历史记录
        history = proxy.load_intent_history(test_user_id)
        print(f"\n用户历史记录数量: {len(history)}")

        if len(history) == 0:
            print("⚠️  该用户没有历史记录，无法测试检索功能")
            print("建议: 在 web 界面添加一些意图判断历史后再测试")
            return True

        print("\n开始情景匹配检索...")
        top_matches, rankings = proxy.find_similar_intent_history_with_embedding(
            context=test_context, user_id=test_user_id, recall_k=20, top_k=1
        )

        print(f"\n✅ 检索成功，找到 {len(top_matches)} 条相似记录:")
        print("-" * 60)

        for idx, item in enumerate(top_matches, 1):
            print(f"\n记录 {idx}:")
            print(f"  情景: {item.get('context', '无')}")
            print(f"  意图: {item.get('intent', '无')}")
            print(
                f"  相似度: {item.get('similarity_score', 0):.4f} ({item.get('similarity_score', 0) * 100:.1f}%)"
            )

        print("\n完整相似度榜单（前5条）:")
        for idx, item in enumerate(rankings[:5], 1):
            print(f"  Top{idx}: {item.get('context', '无')} -> {item.get('similarity_score', 0):.4f}")

        return True

    except Exception as e:
        print(f"\n❌ 检索失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n🔍 Embedding 检索功能测试\n")

    # 测试 1: 模型加载
    if not test_models_loading():
        print("\n❌ 模型加载测试失败，请检查安装")
        return

    # 测试 2: 检索功能
    if not test_retrieval():
        print("\n❌ 检索功能测试失败")
        return

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！Embedding 检索功能正常工作")
    print("=" * 60)
    print("\n你现在可以启动 web 应用并使用该功能了:")
    print("  python app.py")


if __name__ == "__main__":
    main()
