"""
基于字符 Unicode 范围的中日韩英语言检测器
通过分析字符的 Unicode 编码范围来判断文本语言
"""
import re
from typing import Dict
from language_detectors.base_language_detector import BaseLanguageDetector


class CJKEDetector(BaseLanguageDetector):
    """
    中日韩英语言检测器
    支持检测：中文(简体/繁体)、日语、韩语、英语
    """
    
    # Unicode 范围定义
    CHINESE_RANGES = [
        (0x4E00, 0x9FFF),   # CJK统一汉字
        (0x3400, 0x4DBF),   # CJK扩展A
        (0x20000, 0x2A6DF), # CJK扩展B
        (0x2A700, 0x2B73F), # CJK扩展C
        (0x2B740, 0x2B81F), # CJK扩展D
        (0x2B820, 0x2CEAF), # CJK扩展E
        (0xF900, 0xFAFF),   # CJK兼容汉字
    ]
    
    JAPANESE_RANGES = [
        (0x3040, 0x309F),   # 平假名
        (0x30A0, 0x30FF),   # 片假名
        (0x31F0, 0x31FF),   # 片假名音标扩展
    ]
    
    KOREAN_RANGES = [
        (0xAC00, 0xD7AF),   # 韩文音节
        (0x1100, 0x11FF),   # 韩文字母
        (0x3130, 0x318F),   # 韩文兼容字母
        (0xA960, 0xA97F),   # 韩文字母扩展A
        (0xD7B0, 0xD7FF),   # 韩文字母扩展B
    ]
    
    def __init__(self):
        """初始化检测器"""
        # 编译正则表达式以提高性能
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.number_pattern = re.compile(r'[0-9]')
        # 半角标点符号
        self.punctuation_pattern = re.compile(r'[,.\-!?;:()\[\]{}\'"<>/\\|`~@#$%^&*+=_]')
        # 全角标点符号（包括中日韩常用标点）
        self.fullwidth_punctuation_pattern = re.compile(
            r'[，。、；：！？""''（）《》【】「」『』〈〉〔〕｛｝〖〗…—·～￥％＃＠＆＊＋－＝／＼｜＜＞]'
        )
        self.whitespace_pattern = re.compile(r'\s')
    
    def _is_in_ranges(self, char: str, ranges: list) -> bool:
        """
        检查字符是否在指定的 Unicode 范围内
        
        Args:
            char: 单个字符
            ranges: Unicode 范围列表
        
        Returns:
            布尔值，表示字符是否在范围内
        """
        if not char:
            return False
        code_point = ord(char)
        return any(start <= code_point <= end for start, end in ranges)
    
    def _count_char_types(self, text: str) -> Dict[str, int]:
        """
        统计文本中各类字符的数量
        
        Args:
            text: 要分析的文本
        
        Returns:
            包含各类字符计数的字典
        """
        counts = {
            'chinese': 0,
            'japanese': 0,
            'korean': 0,
            'english': 0,
            'number': 0,
            'punctuation': 0,
            'whitespace': 0,
            'other': 0
        }
        
        for char in text:
            if self.whitespace_pattern.match(char):
                counts['whitespace'] += 1
            elif self.number_pattern.match(char):
                counts['number'] += 1
            elif self.punctuation_pattern.match(char) or self.fullwidth_punctuation_pattern.match(char):
                counts['punctuation'] += 1
            elif self._is_in_ranges(char, self.KOREAN_RANGES):
                counts['korean'] += 1
            elif self._is_in_ranges(char, self.JAPANESE_RANGES):
                counts['japanese'] += 1
            elif self._is_in_ranges(char, self.CHINESE_RANGES):
                counts['chinese'] += 1
            elif self.english_pattern.match(char):
                counts['english'] += 1
            else:
                counts['other'] += 1
        
        return counts
    
    def detect(self, text: str) -> Dict[str, any]:
        """
        同步检测文本的语言
        
        Args:
            text: 要检测的文本
        
        Returns:
            包含语言信息的字典：
            {
                'language': 语言代码 ('zh-cn', 'ja', 'ko', 'en'),
                'confidence': 置信度 (0.0-1.0)
            }
        """
        if not text or not text.strip():
            return {'language': 'unknown', 'confidence': 0.0}
        
        # 统计各类字符
        counts = self._count_char_types(text)
        
        # 计算有效字符总数（排除标点、空白和数字）
        total_chars = (counts['chinese'] + counts['japanese'] + 
                      counts['korean'] + counts['english'])
        
        if total_chars == 0:
            return {'language': 'unknown', 'confidence': 0.0}
        
        # 计算各语言的比例
        chinese_ratio = counts['chinese'] / total_chars
        japanese_ratio = counts['japanese'] / total_chars
        korean_ratio = counts['korean'] / total_chars
        english_ratio = counts['english'] / total_chars
        
        # 判断主要语言
        # 日语优先
        if japanese_ratio >= 0.2:
            # 日语中也可能包含汉字
            confidence = japanese_ratio + chinese_ratio
            return {
                'language': 'ja',
                'confidence': confidence
            }
        
        # 韩语判定
        if korean_ratio >= 0.3:
            return {
                'language': 'ko',
                'confidence': korean_ratio
            }
        
        # 中文判定
        if chinese_ratio >= 0.3:
            return {
                'language': 'zh',
                'confidence': chinese_ratio
            }
        
        # 英语判定
        if english_ratio >= 0.5:
            return {
                'language': 'en',
                'confidence': english_ratio
            }
        
        # 混合语言情况：选择占比最大的
        max_ratio, language = max(
            (chinese_ratio, 'zh'),
            (japanese_ratio, 'ja'),
            (korean_ratio, 'ko'),
            (english_ratio, 'en'),
            key=lambda x: x[0]
        )
        return {'language': language, 'confidence': max_ratio}
    
    async def detect_async(self, text: str) -> Dict[str, any]:
        """
        异步检测文本的语言
        
        Args:
            text: 要检测的文本
        
        Returns:
            包含语言信息的字典
        """
        # 由于检测过程不涉及 I/O 操作，直接调用同步方法
        return self.detect(text)
    
    def get_detailed_analysis(self, text: str) -> Dict[str, any]:
        """
        获取详细的语言分析结果
        
        Args:
            text: 要分析的文本
        
        Returns:
            详细的分析结果，包含各类字符统计和比例
        """
        counts = self._count_char_types(text)
        total_chars = sum(counts.values())
        
        # 计算比例
        ratios = {k: v / total_chars if total_chars > 0 else 0.0 
                 for k, v in counts.items()}
        
        detection_result = self.detect(text)
        
        return {
            'text_length': len(text),
            'total_chars': total_chars,
            'counts': counts,
            'ratios': ratios,
            'detection_result': detection_result
        }


# 测试代码
if __name__ == "__main__":
    detector = CJKEDetector()
    
    # 测试用例
    test_cases = [
        "你好世界",                           # 中文
        "你好，世界！",                        # 中文（带全角标点）
        "こんにちは世界",                      # 日语（平假名+汉字）
        "こんにちは、世界！",                  # 日语（带全角标点）
        "안녕하세요",                          # 韩语
        "Hello World",                       # 英语
        "Hello, World!",                     # 英语（带半角标点）
        "这是一个测试 This is a test",        # 中英混合
        "日本語のテスト",                      # 日语（汉字+平假名+片假名）
        "한국어 테스트",                       # 韩语
        "Mixed 混合 언어 language テスト",    # 多语言混合
        "「こんにちは」《你好》",               # 全角引号和书名号
        "",                                   # 空字符串
        "12345 !@#$%",                       # 只有数字和符号
        "１２３４５！@＃￥％",                 # 全角数字和符号
    ]
    
    print("=" * 80)
    print("CJKE 语言检测器测试")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect(text)
        print(f"\n测试 {i}: {text[:30]}{'...' if len(text) > 30 else ''}")
        print(f"  检测结果: {result['language']}")
        print(f"  置信度: {result['confidence']:.2%}")
        
        # 显示详细分析
        analysis = detector.get_detailed_analysis(text)
        print(f"  字符统计: 中文={analysis['counts']['chinese']}, "
              f"日文={analysis['counts']['japanese']}, "
              f"韩文={analysis['counts']['korean']}, "
              f"英文={analysis['counts']['english']}")
    
    print("\n" + "=" * 80)
