import streamlit as st
import pandas as pd
import spacy
import requests
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import nltk

# 必要なデータをダウンロード
nltk.download('wordnet')
nltk.download('omw-1.4')

# レンマタイザの初期化
lemmatizer = WordNetLemmatizer()

# SpaCyの英語モデルをロード
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# CSVファイルのURLリスト
CSV_URLS = {
    "小学校": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/vocabdata_E.csv",
    "中学校": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/vocabdata_J.csv",
    "高等学校英語コミュニケーション": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/vocabdata_HE.csv",
    "高等学校論理表現": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/vocabdata_HL.csv",
}

TEXT_URLS = {
    "小学校": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/ES.txt",
    "中学校": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/JHS.txt",
    "高等学校英語コミュニケーション": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/EC.txt",
    "高等学校論理表現": "http://hirosakieigo.weblike.jp/appdvlp/txtbk/LE.txt",
}

# データをロードする関数
@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {url}\nエラー: {str(e)}")
        return pd.DataFrame()  # 空のDataFrameを返す

# データのロード
dataframes = {key: load_data(url) for key, url in CSV_URLS.items()}

st.title("TeachLex Scope")
st.markdown("""<p style="font-size:16px;">小学校から高等学校の英語の教科書の使用状況をお知らせします。</p>""", unsafe_allow_html=True)

word = st.text_input("単語を入力してください", "")

# 単語のlemmaを取得する関数
def get_lemma(input_word):
    doc = nlp(input_word)
    return doc[0].lemma_ if doc else input_word  # 最初の単語のlemmaを返す

if word:
    lemma_word = get_lemma(word)
    st.write(f"入力された単語: {word} → lemma形式: {lemma_word}")

    for category, df in dataframes.items():
        if df.empty:
            continue
        result = df[df["単語"] == lemma_word]
        if not result.empty:
            st.subheader(f"{category}の教科書の使用状況")

            if category == "中学校" and "頻度" in result.columns:
                st.markdown(f"<p style='font-weight:normal;'>頻度: <b style='font-size:18px;'>{int(result['頻度'].values[0])}</b></p>", unsafe_allow_html=True)
            elif "頻度" in result.columns and category != "中学校":
                st.markdown(f"<p style='font-weight:normal;'>頻度: <b style='font-size:18px;'>{result['頻度'].values[0]}</b></p>", unsafe_allow_html=True)

            if category in ["小学校", "中学校"] and "語彙レベル" in result.columns:
                st.markdown(f"<p style='font-weight:normal;'>語彙レベル: <b style='font-size:18px;'>{result['語彙レベル'].values[0]}</b></p>", unsafe_allow_html=True)

            if category in ["高等学校英語コミュニケーション", "高等学校論理表現"]:
                if "ARF" in result.columns:
                    st.markdown(f"<p style='font-weight:normal;'>頻度: <b style='font-size:18px;'>{int(result['ARF'].values[0])}</b></p>", unsafe_allow_html=True)
                if "語彙レベル" in result.columns:
                    st.markdown(f"<p style='font-weight:normal;'>語彙レベル: <b style='font-size:18px;'>{result['語彙レベル'].values[0]}</b></p>", unsafe_allow_html=True)
                if "使用教科書数" in result.columns:
                    total_books = 24 if category == "高等学校英語コミュニケーション" else 18
                    st.markdown(f"<p style='font-weight:normal;'>使用教科書数: <b style='font-size:18px;'>{int(result['使用教科書数'].values[0])}/{total_books}種類</b></p>", unsafe_allow_html=True)

            if category in ["小学校", "中学校"]:
                textbooks = ["BS", "HWG", "NH", "NC", "OW", "SS"]
                states = [int(result[book].values[0]) if book in result.columns and result[book].values[0] != 0 else "×" for book in textbooks]
                data_table = pd.DataFrame([textbooks, states], index=["教科書名", "初出の学年"])
                st.table(data_table)

            response = requests.get(TEXT_URLS[category])
            if response.status_code == 200:
                text = response.text
                words = [lemmatizer.lemmatize(word.lower()) for word in text.split()]
                context_words = []
                for idx, w in enumerate(words):
                    if w == lemma_word:
                        start = max(0, idx - 3)
                        end = min(len(words), idx + 4)
                        context_words.extend(words[start:idx] + words[idx+1:end])
                context_text = " ".join(context_words)
                if context_text:
                    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{category}のワードクラウド</p>", unsafe_allow_html=True)
                    word_cloud = WordCloud(width=400, height=200, background_color='white', max_words=200).generate(context_text)
                    st.image(word_cloud.to_array(), width=600)  # テーブルと同じ幅で表示
                else:
                    st.warning(f"'{word}' に関連する語が見つかりませんでした（{category}）。")
            else:
                st.error(f"テキストデータを取得できませんでした: {TEXT_URLS[category]}")

            st.markdown("---")
        else:
            st.warning(f"入力された単語は{category}のリストに含まれていません。")
else:
    st.info("単語を入力すると、結果やワードクラウドが表示されます。")
