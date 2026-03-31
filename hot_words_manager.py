"""
热词管理模块
负责加载、管理和创建热词表，供语音识别使用
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
from dashscope.audio.asr import VocabularyService
import logging
from resource_path import get_resource_path, get_user_data_path, ensure_dir

logger = logging.getLogger(__name__)


class HotWordsManager:
    """热词管理器"""
    
    # ============ 配置常量 ============
    # 要加载的语言列表（对应 hot_words 目录下的文件名）
    ENABLED_LANGUAGES = ['zh-cn', 'en', 'ja']
    
    # 热词权重（1-5之间，推荐3-4）
    DEFAULT_WEIGHT = 4
    
    # 热词表前缀（仅允许数字和小写字母，小于10个字符）
    VOCABULARY_PREFIX = 'vrcasr'
    
    # 热词目录路径
    HOT_WORDS_DIR = 'hot_words'
    HOT_WORDS_PRIVATE_DIR = 'hot_words_private'
    # ================================
    
    # 语言代码映射（文件名 -> DashScope 语言代码）
    LANG_CODE_MAP = {
        'zh-cn': 'zh',
        'zh-tw': 'zh',
        'zh': 'zh',
        'en': 'en',
        'ja': 'ja',
        'ko': 'ko',
        'es': 'es',
        'fr': 'fr',
        'de': 'de',
        'ru': 'ru',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化热词管理器
        
        Args:
            api_key: DashScope API Key（如果为 None，则从环境变量获取）
        """
        self.api_key = api_key
        self.vocabulary_service = VocabularyService(api_key=api_key) if api_key else VocabularyService()
        self.vocabulary_id = None
        self.hot_words = []
        
        # 启动时自动清理旧的热词表
        self._cleanup_old_vocabularies()
    
    def _cleanup_old_vocabularies(self):
        """清理所有使用当前前缀的旧热词表"""
        try:
            logger.info(f"检查是否有前缀为 '{self.VOCABULARY_PREFIX}' 的旧热词表...")
            
            # 列出所有使用该前缀的热词表
            old_vocabularies = self.vocabulary_service.list_vocabularies(
                prefix=self.VOCABULARY_PREFIX,
                page_index=0,
                page_size=100
            )
            
            if old_vocabularies:
                logger.info(f"找到 {len(old_vocabularies)} 个旧热词表，正在清理...")
                
                # 删除所有旧热词表
                for vocab in old_vocabularies:
                    vocab_id = vocab.get('vocabulary_id')
                    try:
                        self.vocabulary_service.delete_vocabulary(vocabulary_id=vocab_id)
                        logger.info(f"已删除旧热词表: {vocab_id}")
                    except Exception as e:
                        logger.warning(f"删除热词表 {vocab_id} 失败: {e}")
                
                logger.info(f"清理完成，共删除 {len(old_vocabularies)} 个旧热词表")
            else:
                logger.info("没有找到需要清理的旧热词表")
        
        except Exception as e:
            logger.warning(f"清理旧热词表时出错: {e}")
            # 不抛出异常，允许程序继续运行
    
    def load_hot_words_from_file(self, file_path: str, lang_code: str) -> List[Dict]:
        """
        从文件加载热词
        
        Args:
            file_path: 热词文件路径
            lang_code: 语言代码（如 'zh', 'en'）
        
        Returns:
            热词列表
        """
        hot_words = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # 忽略空行和注释
                        hot_words.append({
                            'text': word,
                            'weight': self.DEFAULT_WEIGHT,
                            'lang': lang_code
                        })
            
            logger.info(f"从 {file_path} 加载了 {len(hot_words)} 个热词")
            return hot_words
        
        except FileNotFoundError:
            logger.warning(f"热词文件不存在: {file_path}")
            return []
        except Exception as e:
            logger.error(f"加载热词文件失败 {file_path}: {e}")
            return []
    
    def load_all_hot_words(self, base_dir: Optional[str] = None) -> List[Dict]:
        """
        加载所有启用语言的热词
        
        Args:
            base_dir: 热词目录基础路径（如果为 None，使用默认路径）
        
        Returns:
            合并后的热词列表
        """
        # 使用资源路径管理函数获取路径
        hot_words_dir = get_resource_path(self.HOT_WORDS_DIR)
        hot_words_private_dir = get_user_data_path(self.HOT_WORDS_PRIVATE_DIR)
        
        # 确保私人热词目录存在
        ensure_dir(hot_words_private_dir)

        all_hot_words = []
        
        for lang_file in self.ENABLED_LANGUAGES:
            # 获取 DashScope 语言代码
            lang_code = self.LANG_CODE_MAP.get(lang_file.lower(), lang_file)
            
            # 加载公共和私有热词文件
            words = []
            file_path = os.path.join(hot_words_dir, f"{lang_file}.txt")
            if os.path.exists(file_path):
                words += self.load_hot_words_from_file(file_path, lang_code)
            file_path = os.path.join(hot_words_private_dir, f"{lang_file}.txt")
            if os.path.exists(file_path):
                words += self.load_hot_words_from_file(file_path, lang_code)
            
            all_hot_words.extend(words)
        
        self.hot_words = all_hot_words
        logger.info(f"总共加载了 {len(all_hot_words)} 个热词（来自 {len(self.ENABLED_LANGUAGES)} 种语言）")
        
        return all_hot_words
    
    def create_vocabulary(self, target_model: str = 'paraformer-realtime-v2') -> str:
        """
        创建热词表
        
        Args:
            target_model: 目标模型名称
        
        Returns:
            热词表 ID
        """
        if not self.hot_words:
            logger.warning("热词列表为空，请先调用 load_all_hot_words()")
            return None
        
        # 检查热词数量限制（最多500个）
        if len(self.hot_words) > 500:
            logger.warning(f"热词数量 ({len(self.hot_words)}) 超过限制 (500)，将只使用前 500 个")
            vocabulary = self.hot_words[:500]
        else:
            vocabulary = self.hot_words
        
        try:
            # 创建热词表
            self.vocabulary_id = self.vocabulary_service.create_vocabulary(
                prefix=self.VOCABULARY_PREFIX,
                target_model=target_model,
                vocabulary=vocabulary
            )
            
            logger.info(f"成功创建热词表，ID: {self.vocabulary_id}")
            logger.info(f"热词表包含 {len(vocabulary)} 个热词")
            
            return self.vocabulary_id
        
        except Exception as e:
            logger.error(f"创建热词表失败: {e}")
            raise
    
    def update_vocabulary(self, vocabulary_id: Optional[str] = None) -> None:
        """
        更新热词表
        
        Args:
            vocabulary_id: 热词表 ID（如果为 None，使用当前的 vocabulary_id）
        """
        if vocabulary_id is None:
            vocabulary_id = self.vocabulary_id
        
        if vocabulary_id is None:
            raise ValueError("未指定热词表 ID")
        
        if not self.hot_words:
            logger.warning("热词列表为空，请先调用 load_all_hot_words()")
            return
        
        # 检查热词数量限制
        if len(self.hot_words) > 500:
            logger.warning(f"热词数量 ({len(self.hot_words)}) 超过限制 (500)，将只使用前 500 个")
            vocabulary = self.hot_words[:500]
        else:
            vocabulary = self.hot_words
        
        try:
            self.vocabulary_service.update_vocabulary(
                vocabulary_id=vocabulary_id,
                vocabulary=vocabulary
            )
            logger.info(f"成功更新热词表 {vocabulary_id}")
        
        except Exception as e:
            logger.error(f"更新热词表失败: {e}")
            raise
    
    def delete_vocabulary(self, vocabulary_id: Optional[str] = None) -> None:
        """
        删除热词表
        
        Args:
            vocabulary_id: 热词表 ID（如果为 None，使用当前的 vocabulary_id）
        """
        if vocabulary_id is None:
            vocabulary_id = self.vocabulary_id
        
        if vocabulary_id is None:
            raise ValueError("未指定热词表 ID")
        
        try:
            self.vocabulary_service.delete_vocabulary(vocabulary_id=vocabulary_id)
            logger.info(f"成功删除热词表 {vocabulary_id}")
            
            if vocabulary_id == self.vocabulary_id:
                self.vocabulary_id = None
        
        except Exception as e:
            logger.error(f"删除热词表失败: {e}")
            raise
    
    def list_vocabularies(self, prefix: Optional[str] = None) -> List[Dict]:
        """
        列出所有热词表
        
        Args:
            prefix: 热词表前缀过滤（如果为 None，使用默认前缀）
        
        Returns:
            热词表列表
        """
        if prefix is None:
            prefix = self.VOCABULARY_PREFIX
        
        try:
            vocabularies = self.vocabulary_service.list_vocabularies(
                prefix=prefix,
                page_index=0,
                page_size=100
            )
            logger.info(f"找到 {len(vocabularies)} 个热词表")
            return vocabularies
        
        except Exception as e:
            logger.error(f"列出热词表失败: {e}")
            raise
    
    def query_vocabulary(self, vocabulary_id: Optional[str] = None) -> Dict:
        """
        查询热词表详情
        
        Args:
            vocabulary_id: 热词表 ID（如果为 None，使用当前的 vocabulary_id）
        
        Returns:
            热词表详情
        """
        if vocabulary_id is None:
            vocabulary_id = self.vocabulary_id
        
        if vocabulary_id is None:
            raise ValueError("未指定热词表 ID")
        
        try:
            details = self.vocabulary_service.query_vocabulary(vocabulary_id=vocabulary_id)
            logger.info(f"查询热词表 {vocabulary_id} 成功")
            return details
        
        except Exception as e:
            logger.error(f"查询热词表失败: {e}")
            raise
    
    def get_vocabulary_id(self) -> Optional[str]:
        """获取当前热词表 ID"""
        return self.vocabulary_id
    
    def get_hot_words(self) -> List[Dict]:
        """获取当前加载的热词列表"""
        return self.hot_words
    
    def print_hot_words_summary(self):
        """打印热词统计信息"""
        if not self.hot_words:
            print("未加载任何热词")
            return
        
        # 按语言统计
        lang_stats = {}
        for word in self.hot_words:
            lang = word.get('lang', 'unknown')
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
        
        print("=" * 60)
        print("热词统计信息")
        print("=" * 60)
        print(f"总热词数: {len(self.hot_words)}")
        print(f"默认权重: {self.DEFAULT_WEIGHT}")
        print("\n各语言热词数量:")
        for lang, count in sorted(lang_stats.items()):
            print(f"  {lang}: {count} 个")
        print("=" * 60)


# 便捷函数
def create_hot_words_vocabulary(
    target_model: str = 'paraformer-realtime-v2',
    api_key: Optional[str] = None
) -> str:
    """
    便捷函数：加载热词并创建热词表
    
    Args:
        target_model: 目标模型名称
        api_key: DashScope API Key（如果为 None，则从环境变量获取）
    
    Returns:
        热词表 ID
    """
    manager = HotWordsManager(api_key=api_key)
    manager.load_all_hot_words()
    manager.print_hot_words_summary()
    vocabulary_id = manager.create_vocabulary(target_model=target_model)
    return vocabulary_id


# 测试代码
if __name__ == "__main__":
    import dashscope
    from dotenv import load_dotenv
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化 API Key
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
    
    print("\n" + "=" * 60)
    print("热词管理器测试")
    print("=" * 60)
    
    # 创建管理器
    manager = HotWordsManager()
    
    # 加载热词
    print("\n1. 加载热词...")
    manager.load_all_hot_words()
    manager.print_hot_words_summary()
    
    # 显示前10个热词示例
    print("\n热词示例（前10个）:")
    for i, word in enumerate(manager.get_hot_words()[:10], 1):
        print(f"  {i}. {word['text']} (语言: {word['lang']}, 权重: {word['weight']})")
    
    # 创建热词表
    print("\n2. 创建热词表...")
    try:
        vocabulary_id = manager.create_vocabulary(target_model='paraformer-realtime-v2')
        print(f"✓ 热词表创建成功，ID: {vocabulary_id}")
        
        # 查询热词表
        print("\n3. 查询热词表详情...")
        details = manager.query_vocabulary()
        print(f"  创建时间: {details.get('gmt_create')}")
        print(f"  目标模型: {details.get('target_model')}")
        print(f"  状态: {details.get('status')}")
        print(f"  热词数量: {len(details.get('vocabulary', []))}")
        
        # 列出所有热词表
        print("\n4. 列出所有热词表...")
        vocabularies = manager.list_vocabularies()
        print(f"  找到 {len(vocabularies)} 个热词表")
        for vocab in vocabularies:
            print(f"    - {vocab['vocabulary_id']} (创建于: {vocab['gmt_create']})")
        
        # 清理：删除测试创建的热词表
        print("\n5. 清理测试热词表...")
        confirm = input(f"是否删除热词表 {vocabulary_id}? (y/n): ")
        if confirm.lower() == 'y':
            manager.delete_vocabulary()
            print("✓ 热词表已删除")
        else:
            print("保留热词表")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
