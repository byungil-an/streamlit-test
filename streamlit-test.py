import streamlit as st
import openai
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import tiktoken
import datetime
import re
import urllib.parse
import requests
# read_sourcefile_code.py から直接関数をインポート
from read_sourcefile_code import (
    read_ipynb, 
    extract_code_and_comments, 
    read_txtfile
)

#UI構成
title = st.title("データサイエンス自習補助ツール")
openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
pick_txt = st.markdown("**実行機能を選択してください**")
choice_solution=st.selectbox("pick one",["1.類似ソース要約","2.ソースコードフィードバック"])
up_file = st.file_uploader("分析するソースファイルをアップロードしてください(2.ソースコードフィードバック選択時のみ)", type=["ipynb","py","txt","R"])

# 1~10 のリストを作成
options = list(range(1, 11))
# デフォルトを 3 に設定 
ref_number = st.selectbox("検索する参考コードの数を選択してください。", options, index=2)

# overview
overview_txt = st.text_area("分析するソースのoverviewを入力してください。")

#スコア入力欄
myscore, aimedscore = st.columns(2)
# 左側の入力欄
with myscore:
    self_score = st.number_input(
        "自身のスコアを入力してください(2.ソースコードフィードバック選択時のみ)",
        min_value=0.0,  # 最小値を小数に設定
        max_value=100.0,  # 最大値を小数に設定
        value=0.0,  # デフォルト値を0.00000に設定
        step=0.00001,  # 小数点以下5桁までのステップ
        format="%.5f",  # 小数点以下5桁のフォーマット
        key="self_score"
    )

# 右側の入力欄
with aimedscore:
    target_score = st.number_input(
        "目指すスコアを入力してください(2.ソースコードフィードバック選択時のみ)",
        min_value=0.0,  # 最小値を小数に設定
        max_value=100.0,  # 最大値を小数に設定
        value=0.0,  # デフォルト値を0.00000に設定
        step=0.00001,  # 小数点以下5桁までのステップ
        format="%.5f",  # 小数点以下5桁のフォーマット
        key="target_score"
    )
# プロンプト実行モード
choice_mode=st.selectbox("pick using model",["gpt-4","gpt-4o mini","gpt-4o"])
# 競技タイプ
choice_type=st.selectbox("pick competition type",["Time series","Classification","Regression","Others"])
# 評価方法
choice_eval=st.selectbox("pick evaluation(2.ソースコードフィードバック選択時のみ)",
                        ["Regression Error(MAE,R^2,RMSE,RMSLE)",
                        "Classification",
                        "AUC",
                        "Confusion matrix",
                        "Quadratic weighted kappa",
                        "MAP@K",
                        "Weighted multi-label logarithmic loss",
                        "Others"])

# チェックボックスのタイトルリスト
models = [
    "XGBoost",
    "Lightgbm",
    "Catboost",
    "Randomforest",
    "linearregression",
    "Pytorch",
    "Tensorflow",
    "Keras",
    "SupportVectorMachine",
    "Gradientboost",
    "LLM",
    "Others"
]

# 複数選択の状態を管理するためのリスト
selected_models = []

# チェックボックスを作成
st.markdown("### モデルを選択してください:(2.ソースコードフィードバック選択時のみ)")
for option in models:
    if st.checkbox(option):
        selected_models.append(option)

# 選択肢に応じて表示する内容を決定
prompt_default = ""
if choice_solution == "1.類似ソース要約":
    prompt_default = "Summarize the source code with the following requirements.\n#Requirements\n- Output what is done in the code for EDA, preprocessing, model building, and evaluation methods, respectively.\n- Describe the overall purpose of each of these processes.\n- Indicate at the beginning of the code which part of the code corresponds to each process.\n"
elif choice_solution == "2.ソースコードフィードバック":
    prompt_default = "Compare the user's code with the following Kaggle solutions and provide specific feedback for improvement."

st.markdown("### プロンプトを実行する場合はチェックしてください。")
chk_prompt=st.checkbox("プロンプトを実行する。")

# 選択肢に応じた内容でテキストエリアを表示
prompt_area = st.text_area("実行するプロンプトを入力してください。",prompt_default)

# ボタン
col = st.columns(2)
ex_button=col[0].button("実行")
cl_button=col[1].button("クリア")

# GitHub APIの基本情報
LOCAL_METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata")

# ローカルのmetadataフォルダからJSONファイルを取得する関数
def fetch_all_metadata_local(directory_path):
    metadata_list = []

    # ディレクトリ内のファイルを走査
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  # JSONファイルのみ取得
            file_path = os.path.join(directory_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as json_file:
                    metadata = json.load(json_file)
                    metadata["folder"] = directory_path  # フォルダパスを追加
                    metadata_list.append(metadata)
            except json.JSONDecodeError as e:
                print(f"JSONデコードエラー: {filename}, エラー内容: {e}")
            except Exception as e:
                print(f"ファイル読み込みエラー: {filename}, エラー内容: {e}")

    return metadata_list

# 評価方法と競技タイプが一致する中からoverviewの類似度が高い3つを取得
def find_similar_overview(input_overview, metadata_list):
    # 類似度計算のためのリスト作成
    overviews = [metadata.get("overview", "") for metadata in metadata_list]
    overviews.append(input_overview)  # ユーザ入力をリストに追加
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(overviews)

    # コサイン類似度を計算
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # ファイル名と類似度をペアにして表示
    filenames = [metadata.get("code", "unknown_file") for metadata in metadata_list]
    similarity_results = list(zip(filenames, similarities))

    # 類似度の高い順に並べ替え
    similarity_results = sorted(similarity_results, key=lambda x: x[1], reverse=True)

    # 類似度の高い順に並べ替えて3つ選定
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:ref_number]

    # 選ばれたメタデータを返す
    return [metadata_list[i] for i in top_indices]

# ユーザー入力に基づいて参考コードを選定
def select_references(user_input, metadata_list):
    references = []
    priorities = []
    print("select_references start")
    # 条件ごとに分類
    for metadata in metadata_list:
        conditions_met = []

        # 条件A: 評価方法が同じ
        if metadata["evaluation"].lower() == user_input["evaluation"].lower():
            conditions_met.append("A")

        # 条件B: 競技タイプが同じ
        if metadata["type"].lower() == user_input["type"].lower():
            conditions_met.append("B")

        # metadata["model"]を小文字に変換
        metadata_model_lower = metadata["model"].lower() if isinstance(metadata["model"], str) else ''
        # user_input["user_model"]の各要素を小文字に変換
        user_model_lower = [model.lower() for model in user_input["user_model"]]

        # 条件C: 使用モデルが部分一致
        if any(model in metadata_model_lower for model in user_model_lower):
            conditions_met.append("C")

        # 条件に基づいて分類
        priority = len(conditions_met)
        priorities.append({"metadata": metadata, "priority": priority, "conditions_met": conditions_met})

    # 優先順でソート
    priorities = sorted(priorities, key=lambda x: (-x["priority"], x["metadata"]["relative score"]))

    # 参考ソースを10件に絞り込み
    for ref in priorities:
        references.append(ref["metadata"])
        if len(references) == 10:
            break

    return references

#添付ファイルの中身を文字列で取得
def get_file_strs(file):
    #アップロードファイル
    if not file:
        st.info("ファイルをアップロードしてください。")
        st.stop()
  
    #添付ファイルの読み込み
    # 拡張子を確認
    file_extension = os.path.splitext(file.name)[1].lower()  # ファイルの拡張子を取得（.ipynb, .py,.txt）
    #テキストボックスのプロンプトを実行
    code_str = ""
    if file_extension == ".ipynb" or file_extension == ".R":
        notebook = read_ipynb(file)
        code_str = extract_code_and_comments(notebook)
    elif file_extension == ".py" or file_extension == ".txt":
        code_str = read_txtfile(file)
    else:
        st.write("対応していないファイル形式です。")
        st.stop()
    
    return code_str 

# テキストのトークン数を計算
def count_tokens(text, model):
    """テキストのトークン数を計算"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# ソースコードからコメント行を削除
def remove_comments_from_code(code):
    """ソースコードからコメント行を削除"""
    lines = code.split("\n")
    filtered_lines = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(filtered_lines)

# ソースコード:.ipynb のテキストデータから `# Markdown start` ～ `# Markdown end` の範囲を削除
def remove_comments_from_ipynb(ipynb_content):
    """
    .ipynb のテキストデータから `# Markdown start` ～ `# Markdown end` の範囲を削除
    :param ipynb_content: `.ipynb` をテキスト化したデータ
    :return: Markdown セルを削除したテキスト
    """
    # 🔹 `# Markdown start` ～ `# Markdown end` のブロックを削除
    cleaned_content = re.sub(r"# Markdown start[\s\S]*?# Markdown end\n?", "", ipynb_content)

    return cleaned_content

# ファイルをデバッグ用に保存する関数
def save_debug_file(content,src_name):
    """file_content をデバッグ用に 'src_name.txt' の形式で保存"""
    #timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # 現在時刻をフォーマット
    filename = f"{src_name}.txt"
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"✅ デバッグ用ファイルを保存: {filename}")

# 実行ボタンを押した時のアクション
if ex_button:
    # ✅ 処理開始時刻を記録
    start_time = time.time()

    # 参考コードのメタデータを収集
    metadata_list = fetch_all_metadata_local(LOCAL_METADATA_PATH)

    # 選択肢に応じて表示するプロンプト実行内容を決定
    prompt_default = ""
    if choice_solution == "1.類似ソース要約":

        #ユーザメタデータ
        user_input = {
            "evaluation": choice_eval,
            "type": choice_type,
            "overview": overview_txt
        }

        # 評価方法・競技タイプが一致するメタデータを絞り込む
        filtered_metadata = [
            metadata for metadata in metadata_list
            if metadata.get("evaluation") and metadata.get("type")  # None ではないことを確認
            and metadata["evaluation"].strip().lower() == user_input["evaluation"].strip().lower()  # 余分なスペースを除去
            and metadata["type"].strip().lower() == user_input["type"].strip().lower()
        ]       

        # 評価方法・競技タイプが一致する中で overview の類似度が高い3つを選定
        if len(filtered_metadata) <= ref_number:
            references = filtered_metadata
        else:
            references = find_similar_overview(
                user_input["overview"],
                filtered_metadata
            )       

        indicative_sentence_for_prompt = prompt_area #指示文

    elif choice_solution == "2.ソースコードフィードバック":
        code_str = get_file_strs(up_file) #ソースコード記載内容
        #ユーザメタデータ
        user_input = {
            "user_code": code_str,
            "current_score": self_score,
            "target_score": target_score,
            "evaluation": choice_eval,
            "type": choice_type,
            "user_model": selected_models,
            "overview": overview_txt
        }

        # 参考ソースコードを選定
        selected_references = select_references(user_input, metadata_list)

        # 選定した参考ソースコードから類似度の高いoverviewのソースを選ぶ
        if len(selected_references) <= ref_number:
            references = selected_references
        else:
            references = find_similar_overview(
                user_input["overview"],
                selected_references
            )[:ref_number]

        indicative_sentence_for_prompt = str(prompt_area) + "\n" +"User's current score:\n"+str(user_input['current_score'])+"\nUser's target score:\n"+ str(user_input['target_score'])+"\nUser's evaluation method:"+ str(user_input['evaluation']) + "\nUser's code:"+ str(user_input['user_code']) + "\nPlease compare the reference source code and user's code below and provide feedback, including specific examples of improvements.\nRelevant Kaggle solutions:"#指示文

    # URLの取得
    # クリック可能なリンクを Markdown 形式で作成
    url_markdown = "\n".join([f"- [{ref['url']}]({ref['url']})" for ref in references])

    # Streamlit で表示（リンク付き）
    st.markdown("### Reference Codes")
    st.markdown(url_markdown, unsafe_allow_html=True)

    if chk_prompt:
        # OpenAI APIリクエスト
        try:
            #apiキー入力チェック
            if not openai_api_key:
                st.info("OpenAI API Keyを入力してください。")
                st.stop()
            else:
                openai.api_key = openai_api_key  # 入力されたAPIキーを設定

            output_contents = []
            request_count = 0  # APIリクエスト回数のカウント

            for ref in references:
                prompt = ""
                github_url = ref.get("github link")  # GitHubリンクを取得
                
                # GitHubリンクの検証
                if not github_url or github_url.lower() in ["nan", "none", ""] or not github_url.startswith("http"):
                    print(f"スキップ: 無効なGitHubリンク: {github_url}, ファイル: {ref.get('code', 'unknown')}")
                    continue  # GitHubリンクがない場合はスキップ
                
                try:
                    print(f"🔍 元URL: {github_url}")
                    
                    # GitHub URLの形式を確認し、raw URLに変換
                    if "/blob/" in github_url:
                        print(f"🔍 元URL: {github_url}")
                        
                        # 最も確実な方法: URL文字列の直接置換（エンコーディングを保持）
                        # https://github.com/owner/repo/blob/branch/path → https://raw.githubusercontent.com/owner/repo/branch/path
                        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                        print(f"🔍 直接置換URL: {raw_url}")
                        
                        # URLが正しく変換されているか確認
                        if raw_url == github_url:
                            print("⚠️ URL変換が行われていません！")
                        else:
                            print(f"✅ URL変換成功: {github_url} → {raw_url}")
                        
                        # URLをパースして検証
                        parsed_url = urllib.parse.urlparse(github_url)
                        parsed_raw = urllib.parse.urlparse(raw_url)
                        print(f"🔍 元URL path: {parsed_url.path}")
                        print(f"🔍 raw URL path: {parsed_raw.path}")
                        
                        # パス部分の解析（デバッグ用）
                        path_parts = parsed_url.path.split('/')
                        print(f"🔍 URLパス各部分: {path_parts}")
                        
                        # requestsライブラリを使用して直接ファイルを取得
                        headers = {
                            "Accept": "application/vnd.github.v3.raw",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        }
                        
                        print(f"🔍 リクエストヘッダー: {headers}")
                        print(f"🔍 最終生成URL: {raw_url}")
                        
                        # URL解析情報を取得
                        owner = repo = branch = file_path_parts = None
                        if len(path_parts) >= 6 and path_parts[3] == 'blob':
                            owner = path_parts[1]
                            repo = path_parts[2] 
                            branch = path_parts[4]
                            file_path_parts = path_parts[5:]
                        
                        # URLs to tryリストを設定（シンプルに）
                        urls_to_try = [raw_url]
                        
                        # 代替URLとして、デコード版も試す（念のため）
                        if owner and repo and branch and file_path_parts:
                            try:
                                # ファイルパス部分をデコードしてから再エンコード
                                decoded_path_parts = [urllib.parse.unquote(part) for part in file_path_parts]
                                # 各パートを適切にエンコード
                                encoded_path_parts = [urllib.parse.quote(part, safe='') for part in decoded_path_parts]
                                encoded_file_path = '/'.join(encoded_path_parts)
                                alternative_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{encoded_file_path}"
                                if alternative_raw_url != raw_url:
                                    urls_to_try.append(alternative_raw_url)
                                    print(f"🔍 代替URL（再エンコード版）: {alternative_raw_url}")
                            except Exception as e:
                                print(f"⚠️ 代替URL生成エラー: {e}")
                        
                        file_content = None
                        success = False
                        
                        for i, attempt_url in enumerate(urls_to_try):
                            try:
                                # URLの最終検証と修正
                                parsed_attempt = urllib.parse.urlparse(attempt_url)
                                # スペースが含まれているか、適切にエンコードされていない場合は修正
                                if ' ' in parsed_attempt.path:
                                    # パスの各部分を適切にエンコード
                                    path_parts = parsed_attempt.path.split('/')
                                    encoded_parts = []
                                    for part in path_parts:
                                        if part:  # 空文字列でない場合のみ処理
                                            # まずデコードしてから再エンコード
                                            decoded_part = urllib.parse.unquote(part)
                                            encoded_part = urllib.parse.quote(decoded_part, safe='')
                                            encoded_parts.append(encoded_part)
                                        else:
                                            encoded_parts.append(part)
                                    
                                    fixed_path = '/'.join(encoded_parts)
                                    attempt_url = urllib.parse.urlunparse((
                                        parsed_attempt.scheme,
                                        parsed_attempt.netloc,
                                        fixed_path,
                                        parsed_attempt.params,
                                        parsed_attempt.query,
                                        parsed_attempt.fragment
                                    ))
                                    print(f"🔍 URL修正前: {parsed_attempt.path}")
                                    print(f"🔍 URL修正後: {fixed_path}")
                                    print(f"🔍 URL修正: {attempt_url}")
                                
                                print(f"🔍 試行 {i+1}/{len(urls_to_try)}: {attempt_url}")
                                response = requests.get(attempt_url, headers=headers, timeout=10)
                                print(f"🔍 ステータス: {response.status_code}")
                                
                                if response.status_code == 200:
                                    # .ipynbファイルの場合は適切に処理
                                    if github_url.endswith('.ipynb'):
                                        try:
                                            notebook_data = json.loads(response.text)
                                            file_content = extract_code_and_comments(notebook_data)
                                        except:
                                            file_content = response.text
                                    else:
                                        file_content = response.text
                                    print(f"✅ ファイル取得成功: {ref.get('code', 'unknown')}")
                                    success = True
                                    break
                                else:
                                    print(f"⚠️ HTTP {response.status_code}: {attempt_url}")
                                    # レスポンスの詳細を取得
                                    try:
                                        error_detail = response.text[:500] if response.text else "No response body"
                                        print(f"   エラー詳細: {error_detail}")
                                        # 404エラーの場合はGitHubのエラーメッセージを確認
                                        if response.status_code == 404:
                                            print(f"   → ファイルが存在しない可能性があります: {attempt_url}")
                                    except:
                                        pass
                            except requests.exceptions.RequestException as e:
                                print(f"⚠️ リクエストエラー {attempt_url}: {e}")
                            except Exception as e:
                                print(f"⚠️ 予期しないエラー {attempt_url}: {e}")
                        
                        if not success:
                            # すべてのURLが失敗した場合、詳細な情報を出力
                            print(f"❌ すべてのURL試行が失敗しました:")
                            for i, url in enumerate(urls_to_try):
                                print(f"   試行 {i+1}: {url}")
                            
                            # リポジトリの存在確認
                            if owner and repo:
                                try:
                                    repo_check_url = f"https://api.github.com/repos/{owner}/{repo}"
                                    repo_response = requests.get(repo_check_url, headers=headers, timeout=5)
                                    print(f"🔍 リポジトリ存在確認: {repo_check_url} -> {repo_response.status_code}")
                                    if repo_response.status_code == 404:
                                        print(f"   → リポジトリが存在しません: {owner}/{repo}")
                                    elif repo_response.status_code == 403:
                                        print(f"   → リポジトリアクセスが制限されています（レート制限または認証エラー）")
                                except Exception as e:
                                    print(f"⚠️ リポジトリ確認エラー: {e}")
                            
                            error_msg = f"File not found (404): All URL attempts failed"
                            if owner and repo:
                                error_msg += f" - Repository: {owner}/{repo}"
                            if file_path_parts:
                                error_msg += f", File: {'/'.join(file_path_parts)}"
                            raise Exception(error_msg)
                    else:
                        # フォールバック: URLを直接変換してアクセス
                        try:
                            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                            headers = {
                                "Accept": "application/vnd.github.v3.raw",
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                            }
                            response = requests.get(raw_url, headers=headers, timeout=10)
                            
                            if response.status_code == 200:
                                if github_url.endswith('.ipynb'):
                                    try:
                                        notebook_data = json.loads(response.text)
                                        file_content = extract_code_and_comments(notebook_data)
                                    except:
                                        file_content = response.text
                                else:
                                    file_content = response.text
                            else:
                                raise Exception(f"Failed to fetch file: {response.status_code}")
                        except Exception as e:
                            raise Exception(f"Error processing GitHub URL: {e}")
                        
                except Exception as e:
                    print(f"⚠️ GitHubファイル取得エラー: {e}")
                    print(f"   元URL: {github_url}")
                    print(f"   デコードURL: {urllib.parse.unquote(github_url)}")
                    print(f"   ファイル: {ref.get('code', 'unknown')}")
                    st.warning(f"GitHubファイルの取得に失敗しました: {ref.get('code', 'unknown')} - {str(e)}")
                    continue  # このファイルはスキップして次のファイルに進む

                #プロンプト実行文
                prompt = indicative_sentence_for_prompt + "\n" + file_content
                
                save_debug_file(prompt,ref['code'])

                # 🔹 **トークン数が30,000超える場合はコメントを除外**
                input_tokens = count_tokens(prompt,choice_mode)

                if input_tokens > 30000:
                    print(f"⚠️ 入力トークン数 {input_tokens} が 30,000 を超過 → 比較ファイルのコメント削除")

                    if github_url.endswith(".ipynb"):
                        file_content = remove_comments_from_ipynb(file_content)  # .ipynb の場合
                    else:
                        file_content = remove_comments_from_code(file_content)  # .py / .R の場合

                    #プロンプト実行文再定義
                    prompt = indicative_sentence_for_prompt + "\n" + file_content
                    input_tokens = count_tokens(prompt,choice_mode) # 削除後のトークン数を再計算

                    if input_tokens > 30000:
                        print(f"⚠️ 入力トークン数 {input_tokens} が 30,000 を超過 → 比較ファイルのコメント削除")
                        # 拡張子を確認
                        file_extension = os.path.splitext(up_file.name)[1].lower()  # ファイルの拡張子を取得（.ipynb, .py,.txt）
                        if file_extension == ".ipynb" or file_extension == ".R":
                            code_str = remove_comments_from_ipynb(file_content)  # .ipynb の場合
                        elif file_extension == ".py" or file_extension == ".txt":
                            code_str = remove_comments_from_code(file_content)  # .py / .R の場合

                        #プロンプト実行文再定義
                        prompt = str(prompt_area) + "\n" +"User's current score:\n"+str(user_input['current_score'])+"\nUser's target score:\n"+ str(user_input['target_score'])+"\nUser's evaluation method:"+ str(user_input['evaluation']) + "\nUser's code:"+ code_str + "\nPlease compare the following reference source code with the user code and provide feedback, including specific examples of improvements in the user code.\nRelevant Kaggle solutions:\n"+ file_content#指示文
                        input_tokens = count_tokens(prompt,choice_mode) # 削除後のトークン数を再計算
                        print(f"入力トークン数 {input_tokens}")
                # プロンプト実行文を をデバッグ用に保存**

                print("*****プロンプト実行*****")
                input_tokens = count_tokens(prompt,choice_mode)
                response = openai.chat.completions.create(
                    model=choice_mode,  # 使用するモデル（gpt-4やgpt-3.5-turboなど）
                    messages=[  # メッセージリストの構造
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    #max_tokens=2000  # 出力のトークン数制限
                )

                output_text = response.choices[0].message.content.strip()
                output_tokens = count_tokens(output_text,choice_mode)

                # ✅ **1回の出力ごとに `text_area` に表示**
                st.text_area(f"結果表示 {request_count+1}", value=output_text, height=600, key=f"text_area_{request_count}")

                # ✅ **Input / Output のトークン数を表示**
                st.write(f"📝 **入力トークン数: {input_tokens}**")
                st.write(f"💬 **出力トークン数: {output_tokens}**")

                request_count += 1  # リクエスト回数を増やす

                # ✅ 最後のインデックスでなければ1分間のトークン制限を考慮し、適切なタイミングで待機
                if request_count < len(references):
                    if choice_solution == "1.類似ソース要約" and request_count % 3 == 0:
                        print("⏳ 3回リクエストしたので、1分間待機します...")
                        time.sleep(60)  # 3回ごとに1分待機

                    elif choice_solution == "2.ソースコードフィードバック":
                        print("⏳ 1回リクエストしたので、1分間待機します...")
                        time.sleep(60)  # 1回ごとに1分待機

        except Exception as e:
             st.error(f"APIリクエスト中にエラーが発生しました: {e}")

    # ✅ 処理終了時刻を記録
    end_time = time.time()
    # ✅ 実行時間を計算
    execution_time = end_time - start_time
    # ✅ Streamlit 上に実行時間を表示
    st.write(f"🔹 **処理時間: {execution_time:.2f} 秒**")

