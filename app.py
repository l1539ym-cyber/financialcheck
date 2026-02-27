import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
import io

@st.cache_resource
def load_models():
    us_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    kr_model = pipeline("text-classification", model="snunlp/KR-FinBERT-SC")
    return us_model, kr_model

@st.cache_data
def load_ticker_mapping():
    try:
        return pd.read_csv('ticker_mapping.csv', dtype=str)
    except Exception:
        return pd.DataFrame()

us_analyzer, kr_analyzer = load_models()
df_tickers = load_ticker_mapping()

ticker_to_name = {
    "NVDA": "엔비디아", "TSLA": "테슬라", "AAPL": "애플",
    "005930": "삼성전자", "000660": "SK하이닉스",
    "005380": "현대차"
}

st.set_page_config(layout="wide", page_title="글로벌 금융 감성 분석")

with st.sidebar:
    st.header("🔑 API 설정 (한국장 전용)")
    st.write("한국장(Naver News) 분석을 위해 본인의 네이버 개발자 API 키를 입력해주세요.")
    user_client_id = st.text_input("Naver Client ID", type="password")
    user_client_secret = st.text_input("Naver Client Secret", type="password")
    st.caption("⚠️ 입력하신 키는 서버에 절대 저장되지 않으며, 현재 브라우저 창을 닫으면 즉시 폐기됩니다.")
    st.divider()
    st.write("미국장(Yahoo Finance)은 API 키 없이 즉시 이용 가능합니다.")

st.title("📈 글로벌 금융 뉴스 감성 분석 대시보드")
st.header("1. 주요 거시경제 지표")

def get_index_data(ticker_symbol):
    data = yf.Ticker(ticker_symbol).history(period="5d")
    if len(data) >= 2:
        curr = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        change = curr - prev
        pct_change = (change / prev) * 100
        return curr, change, pct_change
    return None, None, None

indices = {"VIX (공포지수)": "^VIX", "S&P 500": "^GSPC", "나스닥": "^IXIC", "코스피": "^KS11"}
cols = st.columns(4)
for i, (name, symbol) in enumerate(indices.items()):
    curr, change, pct = get_index_data(symbol)
    with cols[i]:
        if curr is not None:
            color = "#3b82f6" if (symbol == "^VIX" and change > 0) or (symbol != "^VIX" and change < 0) else "#ef4444"
            st.markdown(f"<div style='padding: 10px; border: 1px solid #555; border-radius: 8px;'><p style='margin:0; font-size:14px; font-weight:bold;'>{name}</p><h3 style='margin:0; color:{color};'>{curr:,.2f}</h3><p style='margin:0; color:{color}; font-size:14px;'>{change:+,.2f} ({pct:+.2f}%)</p></div>", unsafe_allow_html=True)

st.divider()
st.header("2. 종목 주요 지표 및 뉴스 분석")
market_choice = st.radio("시장을 선택하세요:", ("한국장 (Naver News)", "미국장 (Yahoo Finance)"))
raw_ticker_input = st.text_input("분석할 종목명 또는 티커를 입력하세요 (예: 삼성전자, 005930, AAPL):").strip()

def format_market_cap(mcap, market):
    if pd.isna(mcap) or mcap is None: 
        return "N/A"
    if market == "미국장 (Yahoo Finance)":
        if mcap >= 1e12: return f"${mcap/1e12:.2f}T"
        elif mcap >= 1e9: return f"${mcap/1e9:.2f}B"
        elif mcap >= 1e6: return f"${mcap/1e6:.2f}M"
        else: return f"${mcap:,.0f}"
    else:
        if mcap >= 1e12: return f"{mcap/1e12:.2f}조"
        elif mcap >= 1e8: return f"{mcap/1e8:.2f}억"
        else: return f"{mcap:,.0f}"

if st.button("데이터 분석 실행"):
    if not raw_ticker_input:
        st.warning("종목명이나 티커를 입력해주세요.")
    else:
        converted_ticker = raw_ticker_input.upper()
        
        if not df_tickers.empty:
            match_name = df_tickers[df_tickers['Name'].str.upper() == raw_ticker_input.upper()]
            if not match_name.empty:
                converted_ticker = str(match_name.iloc[0]['Ticker']).strip()
        
        search_ticker = converted_ticker if market_choice == "미국장 (Yahoo Finance)" else f"{converted_ticker}.KS"
        ticker_obj = yf.Ticker(search_ticker)
        hist_data = ticker_obj.history(period="1y") 
        
        if len(hist_data) >= 21: 
            curr_price = hist_data['Close'].iloc[-1]
            prev_1d_price = hist_data['Close'].iloc[-2]
            change_1d = curr_price - prev_1d_price
            pct_1d = (change_1d / prev_1d_price) * 100
            color_1d = "#ef4444" if change_1d > 0 else "#3b82f6"
            
            prev_1y_price = hist_data['Close'].iloc[0]
            change_1y_cum = curr_price - prev_1y_price
            pct_1y_cum = (change_1y_cum / prev_1y_price) * 100
            color_1y_cum = "#ef4444" if change_1y_cum > 0 else "#3b82f6"
            
            daily_returns = hist_data['Close'].pct_change().dropna() * 100
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            color_1y = "#ef4444" if mean_return > 0 else "#3b82f6"
            
            st.markdown(f"### [{converted_ticker}] 핵심 지표 (현재가: {curr_price:,.2f})")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>전일 대비 (1D)</p><h3 style='margin:0; color:{color_1d};'>{change_1d:+,.2f} ({pct_1d:+.2f}%)</h3></div>", unsafe_allow_html=True)
            with m_col2: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>1년 누적 수익률 (1Y)</p><h3 style='margin:0; color:{color_1y_cum};'>{change_1y_cum:+,.2f} ({pct_1y_cum:+.2f}%)</h3></div>", unsafe_allow_html=True)
            with m_col3: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>1년 일일 등락률 평균 (표준편차)</p><h3 style='margin:0; color:{color_1y};'>{mean_return:+.2f}% ({std_return:.2f})</h3></div>", unsafe_allow_html=True)
            
            hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['STD_20'] = hist_data['Close'].rolling(window=20).std()
            hist_data['Upper_Band'] = hist_data['SMA_20'] + (hist_data['STD_20'] * 2)
            hist_data['Lower_Band'] = hist_data['SMA_20'] - (hist_data['STD_20'] * 2)
            
            plot_data = hist_data.tail(60)

            st.markdown("#### 📊 주가 및 볼린저 밴드 (최근 60일)")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='주가'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], line=dict(color='orange', width=1.5), name='20일 평균선'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Upper_Band'], line=dict(color='gray', width=1, dash='dash'), name='상단 밴드 (+2 STD)'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Lower_Band'], line=dict(color='gray', width=1, dash='dash'), name='하단 밴드 (-2 STD)', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("상장 기간이 짧아 데이터를 충분히 불러올 수 없습니다.")

        st.divider()
        st.markdown("### 🏢 기업 펀더멘털 및 재무 정보 (최근 3년)")
        
        mcap, per, pbr, eps = None, None, None, None
        rev_data, ni_data, years, debt_ratio = [], [], [], []
        fin_source_chart = "Yahoo Finance"
        val_source_info = "Yahoo"

        try:
            info = ticker_obj.info
            mcap = info.get('marketCap')
            per = info.get('trailingPE')
            pbr = info.get('priceToBook')
            eps = info.get('trailingEps')
        except Exception:
            pass

        try:
            fin = ticker_obj.financials
            bs = ticker_obj.balance_sheet
            
            if fin.empty or bs.empty or 'Total Revenue' not in fin.index or 'Net Income' not in fin.index:
                raise ValueError("야후 재무제표 알맹이 누락")
                
            years = [str(date)[:4] for date in fin.columns[:3]][::-1]
            rev_data = fin.loc['Total Revenue'][:3][::-1].tolist()
            ni_data = fin.loc['Net Income'][:3][::-1].tolist()
            
            total_debt = bs.loc['Total Debt'][:3][::-1].tolist()
            total_equity = bs.loc['Stockholders Equity'][:3][::-1].tolist()
            debt_ratio = [(d / e) * 100 for d, e in zip(total_debt, total_equity)]
        except Exception as e:
            fin_source_chart = "정보 부족"

        needs_chart = (fin_source_chart == "정보 부족")
        needs_val = (pd.isna(per) or per is None or pd.isna(pbr) or pbr is None)

        if market_choice == "한국장 (Naver News)" and (needs_chart or needs_val):
            try:
                url = f"https://finance.naver.com/item/main.naver?code={converted_ticker}"
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                dfs = pd.read_html(io.StringIO(r.text), encoding='euc-kr')
                
                target_df = None
                for df in dfs:
                    if '매출액' in df.to_string():
                        target_df = df
                        break
                
                if target_df is not None:
                    target_df.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in target_df.columns]
                    target_df.set_index(target_df.columns[0], inplace=True)
                    
                    annual_cols = [c for c in target_df.columns if '연간' in c and '202' in c]
                    if not annual_cols:
                        annual_cols = [c for c in target_df.columns if '202' in c][:4]
                    annual_cols = annual_cols[-3:]
                    
                    def get_naver_row(keyword, multiplier=1):
                        row = target_df[target_df.index.str.contains(keyword, na=False)]
                        if not row.empty:
                            vals = pd.to_numeric(row.iloc[0][annual_cols].astype(str).str.replace(',', ''), errors='coerce')
                            return (vals * multiplier).tolist()
                        return []

                    if needs_chart:
                        years = [str(c).split('_')[1] if '_' in str(c) else str(c) for c in annual_cols]
                        rev_data = get_naver_row('매출액', 100000000) 
                        ni_data = get_naver_row('당기순이익', 100000000)
                        debt_ratio = get_naver_row('부채비율', 1)
                        if rev_data:
                            fin_source_chart = "Naver Finance (우회 크롤링)"

                    if needs_val:
                        def get_naver_val(keyword):
                            row = target_df[target_df.index.str.contains(keyword, na=False)]
                            if not row.empty:
                                val = str(row.iloc[0][annual_cols[-1]]).replace(',', '')
                                return pd.to_numeric(val, errors='coerce')
                            return None

                        if pd.isna(per) or per is None: 
                            per = get_naver_val('PER')
                            val_source_info += "+Naver"
                        if pd.isna(pbr) or pbr is None: pbr = get_naver_val('PBR')
                        if pd.isna(eps) or eps is None: eps = get_naver_val('EPS')
            except Exception as ex:
                pass

        st.caption(f"💡 데이터 출처: 차트({fin_source_chart}) / 지표({val_source_info})")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>시가총액</p><h3 style='margin:0; color:#ffffff;'>{format_market_cap(mcap, market_choice)}</h3></div>", unsafe_allow_html=True)
        with f_col2: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>PER (주가수익비율)</p><h3 style='margin:0; color:#ffffff;'>{f'{per:.2f}배' if pd.notna(per) else 'N/A'}</h3></div>", unsafe_allow_html=True)
        with f_col3: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>PBR (주가순자산비율)</p><h3 style='margin:0; color:#ffffff;'>{f'{pbr:.2f}배' if pd.notna(pbr) else 'N/A'}</h3></div>", unsafe_allow_html=True)
        with f_col4: st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0; font-size:14px;'>EPS (주당순이익)</p><h3 style='margin:0; color:#ffffff;'>{f'{eps:,.0f}원' if pd.notna(eps) else 'N/A'}</h3></div>", unsafe_allow_html=True)

        if fin_source_chart != "정보 부족" and years and rev_data and ni_data:
            st.markdown("##### 📊 요약 재무제표")
            def format_currency_table(val):
                if pd.isna(val): return "N/A"
                if market_choice == "미국장 (Yahoo Finance)": return f"${val/1e9:,.2f}B"
                else: return f"{val/1e12:,.2f}조"
                    
            df_fin_display = pd.DataFrame({
                "매출액": [format_currency_table(v) for v in rev_data],
                "당기순이익": [format_currency_table(v) for v in ni_data],
                "부채비율(%)": [f"{v:,.2f}%" if pd.notna(v) else "N/A" for v in debt_ratio] if debt_ratio else ["N/A"]*len(years)
            }, index=years)
            st.dataframe(df_fin_display.T, use_container_width=True)

            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=years, y=rev_data, name='매출액', marker_color='#3b82f6', yaxis='y1'))
            ni_colors = ["#ef4444" if val > 0 else "#3b82f6" for val in ni_data]
            fig_fin.add_trace(go.Bar(x=years, y=ni_data, name='당기순이익', marker_color=ni_colors, yaxis='y1'))
            if debt_ratio:
                fig_fin.add_trace(go.Scatter(x=years, y=debt_ratio, name='부채비율(%)', mode='lines+markers', marker_color='orange', yaxis='y2'))
            
            fig_fin.update_layout(
                title="3개년 실적 및 재무건전성 추이 차트", template='plotly_dark', barmode='group',
                yaxis=dict(title='금액'), yaxis2=dict(title='부채비율 (%)', overlaying='y', side='right', showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fin, use_container_width=True)
        else:
            st.warning("해당 종목의 3개년 상세 재무제표 데이터를 수집할 수 없습니다.")

        st.divider()
        st.subheader("📌 4. 주요 투자 의견 (컨센서스)")
        try:
            recos = ticker_obj.recommendations
            if recos is not None and not recos.empty:
                latest_reco = recos.iloc[-1]
                s_buy = latest_reco.get('strongBuy', 0)
                buy = latest_reco.get('buy', 0)
                hold = latest_reco.get('hold', 0)
                sell = latest_reco.get('sell', 0)
                s_sell = latest_reco.get('strongSell', 0)
                
                total_analysts = s_buy + buy + hold + sell + s_sell
                if total_analysts > 0:
                    avg_score = ((s_buy * 2) + (buy * 1) + (hold * 0) + (sell * -1) + (s_sell * -2)) / total_analysts
                    reco_color = "#ef4444" if avg_score > 0 else ("#3b82f6" if avg_score < 0 else "#ffffff")
                    st.markdown(f"<div style='padding: 15px; border: 1px solid #555; border-radius: 10px;'><p style='margin:0;'>총 {total_analysts}명 참여 (Strong Buy: {s_buy}, Buy: {buy}, Hold: {hold}, Sell: {sell}, Strong Sell: {s_sell})</p><h3 style='margin:10px 0 0 0; font-size: 20px;'>평균 컨센서스 점수: <span style='color:{reco_color}; font-weight:bold;'>{avg_score:+.2f}점</span></h3></div>", unsafe_allow_html=True)
                st.dataframe(recos.tail(1), use_container_width=True)
            else:
                st.info("해당 종목은 투자 의견 데이터가 제공되지 않습니다.")
        except:
            st.info("투자 의견 데이터를 처리하는 중 오류가 발생했습니다.")

        st.divider()
        st.subheader("📰 5. 뉴스 AI 감성 분석 결과 (최대 10건)")
        
        total_sentiment_score = 0
        analyzed_count = 0

        def process_news_sentiment(title, label, score):
            if label == 'positive': return 1, "#ef4444"
            elif label == 'negative': return -1, "#3b82f6"
            else: return 0, "#ffffff"

        if market_choice == "미국장 (Yahoo Finance)":
            news_list = ticker_obj.news
            if news_list:
                for news in news_list[:10]:
                    title = news.get('title') or news.get('content', {}).get('title') or '제목 없음'
                    if title != '제목 없음':
                        result = us_analyzer(title)[0]
                        label = result['label'].lower()
                        score = result['score'] * 100
                        point, color = process_news_sentiment(title, label, score)
                        
                        total_sentiment_score += point
                        analyzed_count += 1
                        
                        with st.container(border=True):
                            st.write(f"**{title}**")
                            st.markdown(f"🤖 AI 분석: <span style='color:{color}; font-weight:bold;'>{label.upper()}</span> (부여 점수: <span style='color:{color};'>{point}점</span> / 신뢰도: {score:.1f}%)", unsafe_allow_html=True)
                
                score_color = "#ef4444" if total_sentiment_score > 0 else ("#3b82f6" if total_sentiment_score < 0 else "#ffffff")
                st.markdown(f"### 📊 종합 뉴스 감성 점수: <span style='color:{score_color};'>{total_sentiment_score:+}점</span> (총 {analyzed_count}건 분석)", unsafe_allow_html=True)
            else:
                st.error("최신 영문 뉴스를 불러오지 못했습니다.")
                
        elif market_choice == "한국장 (Naver News)":
            search_keyword = ticker_to_name.get(raw_ticker_input, raw_ticker_input) 
            
            if not user_client_id or not user_client_secret:
                st.error("🚨 좌측 사이드바에서 네이버 API 키를 먼저 입력해주세요!")
            else:
                headers = {"X-Naver-Client-Id": user_client_id, "X-Naver-Client-Secret": user_client_secret}
                url = f"https://openapi.naver.com/v1/search/news.json?query={search_keyword}&display=10"
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    items = response.json().get('items', [])
                    for item in items:
                        title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
                        result = kr_analyzer(title)[0]
                        label = result['label'].lower()
                        score = result['score'] * 100
                        point, color = process_news_sentiment(title, label, score)
                        
                        total_sentiment_score += point
                        analyzed_count += 1
                        
                        with st.container(border=True):
                            st.write(f"**{title}**")
                            st.markdown(f"🤖 AI 분석: <span style='color:{color}; font-weight:bold;'>{label.upper()}</span> (부여 점수: <span style='color:{color};'>{point}점</span> / 신뢰도: {score:.1f}%)", unsafe_allow_html=True)
                    
                    score_color = "#ef4444" if total_sentiment_score > 0 else ("#3b82f6" if total_sentiment_score < 0 else "#ffffff")
                    st.markdown(f"### 📊 종합 뉴스 감성 점수: <span style='color:{score_color};'>{total_sentiment_score:+}점</span> (총 {analyzed_count}건 분석)", unsafe_allow_html=True)
                else:
                    st.error("네이버 API 호출 실패. 입력하신 API 키가 정확한지 확인해주세요.")