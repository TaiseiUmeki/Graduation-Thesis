"""
メインエントリーポイント
A→B→C→Dのフロー統合
"""
from pathlib import Path
from typing import Optional

from config import Config
from models.session import Session
from models.constraints import Constraint, ConstraintType
from engines.query_interpreter import QueryInterpreter
from engines.vector_generator import VectorGenerator
from engines.image_generator import ImageGenerator
from utils.openai_client import OpenAIClient
from utils.image_utils import ImageUtils


class TrueCodingSystem:
    """TrueCodingシステムのメインクラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        システムを初期化
        
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
        """
        if api_key:
            Config.set_api_key(api_key)
        
        # クライアントの初期化
        self.client = OpenAIClient()
        
        # エンジンの初期化
        self.query_interpreter = QueryInterpreter(self.client)
        self.vector_generator = VectorGenerator(self.client)
        self.image_generator = ImageGenerator(self.client)
        
        # ユーティリティ
        self.image_utils = ImageUtils()
        
        # セッション
        self.session: Optional[Session] = None
    
    # ========== フェーズA: 入力 ==========
    
    def start_session(self, image_path: str, query: str) -> Session:
        """
        新しいセッションを開始（フェーズA）
        
        Args:
            image_path: 粘土の中間生成物の画像パス
            query: ユーザーのクエリ
        
        Returns:
            セッション
        """
        print("=" * 60)
        print("フェーズA: 入力")
        print("=" * 60)
        
        # 新しいセッションを作成
        self.session = Session()
        self.session.current_phase = "A"
        
        # 画像を保存
        saved_image_path = self.image_utils.save_uploaded_image(
            image_path,
            Config.IMAGES_DIR,
            prefix="initial"
        )
        self.session.initial_image_path = saved_image_path
        print(f"画像を保存しました: {saved_image_path}")
        
        # クエリを記録
        self.session.initial_query = query
        self.session.add_query(query)
        print(f"クエリ: {query}")
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        return self.session
    
    # ========== フェーズB: クエリ解釈 ==========
    
    def interpret_query(
        self,
        additional_image: Optional[str] = None,
        use_image_context: bool = True
    ) -> list:
        """
        クエリを解釈し、複数の解釈案を生成（フェーズB）
        
        Args:
            additional_image: 追加の補足画像（オプション）
            use_image_context: 元画像をコンテキストとして使用するか
        
        Returns:
            解釈案のリスト
        """
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        print("\n" + "=" * 60)
        print("フェーズB: クエリ解釈")
        print("=" * 60)
        
        self.session.current_phase = "B"
        
        # 使用する画像を決定
        context_image = None
        if use_image_context:
            context_image = self.session.initial_image_path
        if additional_image:
            self.session.additional_image_path = additional_image
            context_image = additional_image
        
        # 最新のクエリを取得
        current_query = self.session.query_history[-1]
        
        # 解釈案を生成
        print(f"\nクエリを解釈中: {current_query}")
        if context_image:
            print(f"画像コンテキスト: {context_image}")
        
        interpretations = self.query_interpreter.generate_interpretations(
            query=current_query,
            context_image_path=context_image,
            previous_interpretations=self.session.interpretations
        )
        
        # セッションに追加
        self.session.add_interpretations(interpretations)
        
        # 解釈案を表示
        print(f"\n{len(interpretations)}つの解釈案を生成しました：")
        for interp in interpretations:
            print(f"\n[{interp.id}] {interp.text}")
            print(f"    根拠: {interp.reasoning}")
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        return interpretations
    
    def select_interpretation(self, interpretation_id: int):
        """
        解釈案を選択
        
        Args:
            interpretation_id: 選択する解釈案のID
        """
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        selected = self.session.select_interpretation(interpretation_id)
        if selected:
            print(f"\n解釈案 [{interpretation_id}] を選択しました")
            print(f"{selected.text}")
            self.session.save(Config.SESSIONS_DIR)
        else:
            print(f"警告: ID {interpretation_id} の解釈案が見つかりません")
    
    def refine_query(self, additional_details: str):
        """
        クエリを詳細化して再解釈
        
        Args:
            additional_details: 追加の詳細情報
        """
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        print("\nクエリを詳細化中...")
        
        refined = self.query_interpreter.refine_query(
            self.session.initial_query,
            additional_details,
            self.session.selected_interpretation
        )
        
        print(f"詳細化されたクエリ: {refined}")
        
        # 新しいクエリとして追加
        self.session.add_query(refined)
        self.session.save(Config.SESSIONS_DIR)
    
    # ========== フェーズC: 特徴ベクトル生成 ==========
    
    def generate_vector(self) -> dict:
        """
        選択された解釈から特徴ベクトルを生成（フェーズC）
        
        Returns:
            特徴ベクトル情報
        """
        if not self.session or not self.session.selected_interpretation:
            raise ValueError("解釈案が選択されていません")
        
        print("\n" + "=" * 60)
        print("フェーズC: 特徴ベクトル生成")
        print("=" * 60)
        
        self.session.current_phase = "C"
        
        # 特徴ベクトルを生成
        print("\n特徴ベクトルを生成中...")
        vector = self.vector_generator.generate_vector_from_interpretation(
            self.session.selected_interpretation,
            self.session.get_active_constraints()
        )
        
        self.session.set_vector(vector)
        
        # 主要な属性を表示
        from models.attribute_space import ATTR_SPACE
        top_attrs = ATTR_SPACE.get_top_attributes(vector, top_k=10)
        
        print("\n主要な属性:")
        for attr_key, weight in top_attrs:
            attr_name = ATTR_SPACE.get_attribute_name(attr_key)
            print(f"  {attr_name}: {weight:.2f}")
        
        # 関連属性を検索
        print("\n関連性の高い属性を検索中...")
        related = self.vector_generator.find_related_attributes(
            self.session.selected_interpretation,
            self.session.initial_query,
            top_k=10
        )
        
        self.session.recommended_attributes = [attr_key for attr_key, _, _ in related]
        
        print("\n推薦属性:")
        for attr_key, attr_name, score in related:
            print(f"  {attr_name} (関連度: {score:.2f})")
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        return {
            "vector": vector,
            "top_attributes": top_attrs,
            "recommended_attributes": related
        }
    
    # ========== フェーズD: 画像生成と批評 ==========
    
    def generate_image(self) -> str:
        """
        特徴ベクトルから画像を生成（フェーズD）
        
        Returns:
            生成画像のパス
        """
        if not self.session or not self.session.current_vector:
            raise ValueError("特徴ベクトルが生成されていません")
        
        print("\n" + "=" * 60)
        print("フェーズD: 画像生成")
        print("=" * 60)
        
        self.session.current_phase = "D"
        
        # 元画像の説明を取得
        print("\n元画像を分析中...")
        base_description = self.image_generator.extract_description_from_image(
            self.session.initial_image_path
        )
        print(f"元画像の説明: {base_description}")
        
        # 画像を生成
        print("\n画像を生成中...")
        generated = self.image_generator.generate_from_vector(
            base_description,
            self.session.current_vector,
            self.session.get_active_constraints()
        )
        
        self.session.add_generated_image(generated)
        
        print(f"\n画像を生成しました: {generated.image_path}")
        
        # 制約候補の属性を提案
        suggestions = self.image_generator.suggest_constraint_attributes(
            self.session.current_vector,
            num_suggestions=5
        )
        
        print("\n制約追加の候補属性:")
        for i, (attr_key, attr_name, current_value, group) in enumerate(suggestions, 1):
            print(f"  [{i}] {attr_name} (現在値: {current_value:.2f}, グループ: {group})")
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        return generated.image_path
    
    def add_constraint(
        self,
        attribute_key: str,
        constraint_type: str,
        value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        description: str = ""
    ):
        """
        制約を追加して画像を再生成
        
        Args:
            attribute_key: 属性キー
            constraint_type: 制約タイプ（"less_than", "greater_than", "equal", "range"）
            value: 値（単一値の場合）
            min_value: 最小値（範囲の場合）
            max_value: 最大値（範囲の場合）
            description: 制約の説明
        """
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        print(f"\n制約を追加: {attribute_key}")
        
        # 制約を作成
        constraint = Constraint(
            attribute=attribute_key,
            constraint_type=ConstraintType(constraint_type),
            value=value,
            min_value=min_value,
            max_value=max_value,
            description=description
        )
        
        self.session.add_constraint(constraint)
        print(f"制約: {constraint.to_natural_language()}")
        
        # ベクトルを更新
        print("\n特徴ベクトルを更新中...")
        updated_vector = self.vector_generator.update_vector_with_constraint(
            self.session.current_vector,
            constraint,
            self.session.selected_interpretation
        )
        
        self.session.set_vector(updated_vector)
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        print("特徴ベクトルを更新しました")
    
    def analyze_current_image(self) -> str:
        """
        現在の生成画像を分析
        
        Returns:
            分析結果
        """
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        latest_image = self.session.get_latest_image()
        if not latest_image:
            raise ValueError("生成画像がありません")
        
        print("\n画像を分析中...")
        analysis = self.image_generator.analyze_image_with_gpt(latest_image.image_path)
        
        print("\n【分析結果】")
        print(analysis)
        
        return analysis
    
    def get_session_summary(self) -> dict:
        """
        現在のセッションの要約を取得
        
        Returns:
            セッション情報の辞書
        """
        if not self.session:
            return {"error": "セッションが存在しません"}
        
        return {
            "session_id": self.session.session_id,
            "current_phase": self.session.current_phase,
            "initial_image": self.session.initial_image_path,
            "initial_query": self.session.initial_query,
            "num_queries": len(self.session.query_history),
            "num_interpretations": len(self.session.interpretations),
            "selected_interpretation": self.session.selected_interpretation.text if self.session.selected_interpretation else None,
            "num_generated_images": len(self.session.generated_images),
            "num_constraints": len(self.session.constraints),
            "latest_image": self.session.get_latest_image().image_path if self.session.get_latest_image() else None
        }


def main():
    """デモ実行"""
    print("TrueCoding System - デモ")
    print("=" * 60)
    
    # システムを初期化（APIキーは環境変数から取得）
    system = TrueCodingSystem()
    
    # ========== 以下を編集: 画像パスとクエリを指定 ==========
    image_path = "data/images/original_tank.jpg"  # 粘土画像のパスを指定
    query = "もっとかわいいデザインにしたい"                  # クエリを指定
    
    # フェーズA: セッション開始
    print("\n### フェーズA: 入力 ###")
    session = system.start_session(image_path=image_path, query=query)
    
    # フェーズB: クエリ解釈と詳細化のループ
    print("\n### フェーズB: クエリ解釈 ###")
    while True:
        interpretations = system.interpret_query(use_image_context=True)
        
        # ユーザーに解釈案の選択を促す
        print("\n解釈案を選んでください:")
        while True:
            try:
                choice = input("解釈案の番号を入力 (1-3)、または 'r' でクエリ詳細化: ").strip().lower()
                if choice == 'r':
                    # クエリを詳細化
                    print("\nクエリを詳細化してください:")
                    refined_query = input("詳細化クエリを入力: ").strip()
                    
                    # 参考画像をオプションで追加
                    add_image = input("補足用の画像を追加しますか？ (y/n): ").strip().lower()
                    additional_image = None
                    if add_image == 'y':
                        additional_image = input("画像パスを入力: ").strip()
                    
                    # クエリを詳細化
                    system.refine_query(refined_query)
                    
                    # 再度解釈案を生成するため、ループの最初に戻る
                    print("\n詳細化されたクエリで新しい解釈案を生成します...\n")
                    break
                else:
                    choice_id = int(choice)
                    if 1 <= choice_id <= 3:
                        system.select_interpretation(interpretation_id=choice_id)
                        # 解釈選択成功 - 内側ループを抜ける
                        break
                    else:
                        print("1～3の数字を入力してください")
            except ValueError:
                print("数字で入力するか、'r' でクエリを詳細化してください")
        
        # 解釈が選択された場合（'r'でない場合）、外側ループを抜ける
        if choice != 'r':
            break
    
    
    # フェーズC: ベクトル生成
    print("\n### フェーズC: 特徴ベクトル生成 ###")
    vector_info = system.generate_vector()
    
    # フェーズD: 画像生成
    print("\n### フェーズD: 画像生成 ###")
    image_path = system.generate_image()
    
    print("\n使用方法:")
    print("1. system.start_session(image_path, query) でセッション開始")
    print("2. system.interpret_query() でクエリを解釈")
    print("3. system.select_interpretation(id) で解釈を選択")
    print("4. system.generate_vector() で特徴ベクトル生成")
    print("5. system.generate_image() で画像生成")
    print("6. system.add_constraint(...) で制約追加と再生成")
    print("7. system.analyze_current_image() で画像分析")
    
    print("\n詳細は各メソッドのdocstringを参照してください。")
    
    return system


if __name__ == "__main__":
    system = main()
