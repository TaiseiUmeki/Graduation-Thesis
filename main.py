"""
メインエントリーポイント
A→B→C→Dのフロー統合
"""
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from config import Config
from models.session import Session
from models.constraints import Constraint, ConstraintType
from models.attribute_space import ATTR_SPACE, AttributeVector
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

        # 探索木のルートノードを作成
        if self.session.current_node_id is None:
            self.session.add_root_node(
                vector,
                self.session.constraints,
                note="root"
            )
        
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
        
        # 画像を生成
        print("\n画像を生成中...")
        generated = self.image_generator.generate_from_vector(
            self.session.initial_image_path,
            self.session.current_vector,
            self.session.get_active_constraints()
        )
        
        self.session.add_generated_image(generated)

        # 現在ノードに画像パスを記録
        self.session.update_current_node_image(generated.image_path, generated.prompt)
        
        print(f"\n画像を生成しました: {generated.image_path}")
        
        # 制約候補の属性を提案
        suggestions = self.image_generator.suggest_constraint_attributes(
            self.session.current_vector,
            num_suggestions=10
        )
        
        print("\n制約追加の候補属性:")
        for i, (attr_key, attr_name, current_value, group) in enumerate(suggestions, 1):
            # 属性キーも併記して、入力時に迷わないようにする
            print(
                f"  [{i}] {attr_name} ({attr_key}) "
                f"(現在値: {current_value:.2f}, グループ: {group})"
            )
        
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
        
        # クローズドノードからの斥力を適用
        closed_nodes = self.session.get_closed_nodes()
        if closed_nodes:
            print(f"\nクローズドノードからの斥力を適用中（{len(closed_nodes)}個）...")
            closed_vectors = [node.vector for node in closed_nodes]
            updated_vector = self.vector_generator.apply_repulsion(
                updated_vector,
                closed_vectors,
                min_squared_distance=0.5,  # この値を調整可能
                repulsion_strength=0.3     # この値を調整可能
            )
        
        self.session.set_vector(updated_vector)

        # 探索木に子ノードを追加
        self.session.add_child_node(
            updated_vector,
            self.session.constraints,
            note=f"add_constraint:{attribute_key}"
        )
        
        # セッションを保存
        self.session.save(Config.SESSIONS_DIR)
        
        print("特徴ベクトルを更新しました")

    # ========== フェーズD: ファジー属性調整（LLMマッピング） ==========
    def apply_fuzzy_adjustment(
        self,
        user_text: str,
        max_suggestions: int = 6,
        default_delta: float = 0.25,
        auto_generate_image: bool = False
    ) -> List[Tuple[str, float, str]]:
        """
        自由入力テキストを属性調整にマッピングしてベクトルを更新
        Args:
            user_text: ユーザーの意図（例: "もっとサイバーに"）
            max_suggestions: 最大提案数
            default_delta: 典型的な調整幅（±）
            auto_generate_image: Trueなら更新後に画像も生成
        Returns:
            提案リスト [(attribute_key, delta, reason), ...]
        """
        if not self.session or not self.session.current_vector:
            raise ValueError("特徴ベクトルが生成されていません")

        print("\nファジー調整を実行中...")
        adjustments = self.vector_generator.suggest_attribute_adjustments_from_text(
            user_text,
            self.session.current_vector,
            max_suggestions=max_suggestions,
            default_delta=default_delta
        )

        if not adjustments:
            print("提案がありませんでした")
            return []

        # ベクトルを更新
        updated_weights = dict(self.session.current_vector.weights)
        for attr_key, delta, reason in adjustments:
            before = updated_weights.get(attr_key, 0.0)
            after = float(np.clip(before + delta, -1.0, 1.0))
            updated_weights[attr_key] = after
            attr_name = ATTR_SPACE.get_attribute_name(attr_key) or attr_key
            print(f"  {attr_name}: {before:+.2f} -> {after:+.2f} (delta {delta:+.2f}) {reason if reason else ''}")

        new_vector = AttributeVector(weights=updated_weights)
        self.session.set_vector(new_vector)

        # 探索木に子ノードを追加（メモ付き）
        note = f"fuzzy:{user_text[:30]}" if user_text else "fuzzy"
        self.session.add_child_node(new_vector, self.session.constraints, note=note)
        self.session.save(Config.SESSIONS_DIR)

        print("ベクトルを更新しました（ファジー調整）")

        if auto_generate_image:
            self.generate_image()

        return adjustments
    
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
    
    def search_attributes(
        self,
        user_text: str,
        max_results: int = 5,
        return_expanded: bool = True
    ):
        """
        自由テキストから属性を検索（LLMマッピング）
        
        Args:
            user_text: 検索クエリ（例: "サイバーな"）
            max_results: 最大件数
            return_expanded: Trueなら拡張版、Falseなら簡易版
        
        Returns:
            拡張版: [{"attribute_key": str, "attribute_name": str}, ...]
            簡易版: ["group:key", ...]
        """
        print(f"\n属性を検索中: 「{user_text}」")
        results = self.vector_generator.map_text_to_attributes(
            user_text,
            max_results=max_results,
            return_expanded=return_expanded
        )
        
        if not results:
            print("該当する属性が見つかりませんでした")
        else:
            print(f"\n検索結果 ({len(results)}件):")
            if return_expanded:
                for i, item in enumerate(results, 1):
                    print(f"  [{i}] {item['attribute_name']} ({item['attribute_key']})")
            else:
                for i, key in enumerate(results, 1):
                    name = ATTR_SPACE.get_attribute_name(key) or key
                    print(f"  [{i}] {name} ({key})")
        
        return results
    
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

    # ========== 探索木ユーティリティ ==========

    def get_attribute_value(self, attribute_key: str) -> Optional[float]:
        """
        現在の特徴ベクトルから指定属性の値を取得
        
        Args:
            attribute_key: 属性キー（例: "material:stone", "color:pastel"）
        
        Returns:
            属性値（0.0～1.0）、存在しない場合はNone
        """
        if not self.session or not self.session.current_vector:
            print("特徴ベクトルが存在しません")
            return None
        
        value = self.session.current_vector.weights.get(attribute_key)
        if value is None:
            print(f"属性 '{attribute_key}' が見つかりません")
            # 属性名の候補を表示
            matching = [k for k in self.session.current_vector.weights.keys() if attribute_key.split(':')[-1] in k]
            if matching:
                print(f"類似の属性: {', '.join(matching[:5])}")
            return None
        
        # 属性名も表示
        attr_name = self.session.attribute_space.get_attribute_name(attribute_key)
        if attr_name:
            print(f"{attribute_key} ({attr_name}): {value:.4f}")
        else:
            print(f"{attribute_key}: {value:.4f}")
        
        return value
    
    def show_vector(self, threshold: float = 0.0, top_k: Optional[int] = None):
        """
        現在の特徴ベクトルを表示
        
        Args:
            threshold: 表示する最小絶対値（デフォルト: 0.0）
            top_k: 上位k個のみ表示（指定しない場合は閾値以上すべて）
        """
        if not self.session or not self.session.current_vector:
            print("特徴ベクトルが存在しません")
            return
        
        # 値でソート（絶対値の大きい順）
        sorted_attrs = sorted(
            self.session.current_vector.weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # フィルタリング（絶対値で判定）
        filtered = [(attr, val) for attr, val in sorted_attrs if abs(val) > threshold]
        
        if top_k:
            filtered = filtered[:top_k]
        
        if not filtered:
            print(f"閾値 {threshold} 以上の属性がありません")
            return
        
        print(f"\n現在の特徴ベクトル（{len(filtered)}個の属性）:")
        print("（負値は属性を避ける/逆方向探索を意味します）")
        for attr, val in filtered:
            attr_name = ATTR_SPACE.get_attribute_name(attr)
            if val >= 0:
                indicator = "強調"
            else:
                indicator = "回避"
            
            if attr_name:
                print(f"  {attr:40s} ({attr_name:20s}): {val:7.4f} [{indicator}]")
            else:
                print(f"  {attr:40s}: {val:7.4f} [{indicator}]")

    def list_nodes(self):
        if not self.session:
            return []
        return self.session.list_nodes()

    def get_node_details(self, node_id: int):
        """特定ノードの詳細（制約含む）を表示"""
        if not self.session:
            raise ValueError("セッションが開始されていません")
        
        node = self.session._find_node(node_id)
        if not node:
            print(f"ノード {node_id} が見つかりません")
            return None
        
        print(f"\n{'='*60}")
        print(f"ノード {node_id} の詳細")
        print(f"{'='*60}")
        print(f"親ノード: {node.parent_id}")
        print(f"クローズド: {node.is_closed}")
        print(f"メモ: {node.note}")
        print(f"作成日時: {node.timestamp.isoformat()}")
        
        if node.constraints:
            print(f"\n制約 ({len(node.constraints)}個):")
            for i, c in enumerate(node.constraints, 1):
                print(f"  [{i}] {c.to_natural_language()}")
        else:
            print("\n制約: なし")
        
        if node.generated_image_path:
            print(f"\n生成画像: {node.generated_image_path}")
        else:
            print("\n生成画像: なし")
        
        return node

    def revert_to_node(self, node_id: int):
        if not self.session:
            raise ValueError("セッションが開始されていません")
        self.session.revert_to_node(node_id)
        print(f"ノード {node_id} に戻りました")

    def mark_node_closed(self, node_id: int):
        if not self.session:
            raise ValueError("セッションが開始されていません")
        self.session.mark_closed(node_id)
        print(f"ノード {node_id} をクローズドにしました")


def main():
    """デモ実行"""
    print("TrueCoding System - デモ")
    print("=" * 60)
    
    # システムを初期化（APIキーは環境変数から取得）
    system = TrueCodingSystem()
    
    # ========== 以下を編集: 画像パスとクエリを指定 ==========
    image_path = "data/images/original_tank.jpg"  # 粘土画像のパスを指定
    query = "「シャキーン!」という感じのかっこいいデザインにしたい"                  # クエリを指定
    
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
    print("7. system.apply_fuzzy_adjustment('もっとサイバーに') でファジー調整（自由入力からLLMが属性を推論）")
    print("8. system.search_attributes('サイバーな') で属性検索（LLMが自由テキストから関連属性を推論）")
    print("9. system.analyze_current_image() で画像分析")
    
    print("\n探索木操作:")
    print("- system.list_nodes() でノード一覧表示")
    print("- system.get_node_details(node_id) でノード詳細表示（制約含む）")
    print("- system.revert_to_node(node_id) で過去のノードに戻す")
    print("- system.mark_node_closed(node_id) でノードをクローズド化（斥力適用対象に）")
    print("- system.show_vector(top_k=10) で現在の特徴ベクトル表示")
    
    print("\n詳細は各メソッドのdocstringを参照してください。")
    
    return system


if __name__ == "__main__":
    system = main()
