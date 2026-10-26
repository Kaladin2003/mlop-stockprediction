import streamlit as st
import requests
import os

st.set_page_config(page_title="S&P 500 Υβριδικό Dashboard", page_icon="📈", layout="wide")

st.title("📈 S&P 500 Hybrid Stock Dashboard")
st.markdown("Συνδυασμός **Γενικής Μηχανικής Μάθησης (S&P 500 Model)** και **Live Ανάλυσης Συναισθήματος (NLP)**.")
st.divider()

ticker = st.text_input("Εισάγετε Σύμβολο Μετοχής (π.χ. AAPL, MSFT, KO, NVDA, F):", "AAPL")

if st.button("Εκτέλεση Πλήρους Ανάλυσης"):
    api_url = os.getenv("API_URL", "http://api:8000") + "/predict"
    payload = {"ticker": ticker.upper()}
    
    try:
        with st.spinner(f'Ανάλυση με το S&P 500 generic μοντέλο για {ticker.upper()}...'):
            response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            st.subheader(f"Αποτελέσματα για: **{data['ticker']}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Τελευταία Τιμή", value=f"${data['latest_close_price']}")
            
            with col2:
                st.info(f" **Μοντέλο ML (S&P 500 Generic):**\n\nΠρόβλεψε **{data['prediction_ml']}**\n\n(Σιγουριά: **{data['ml_confidence_percent']}%**)")
                st.caption("Το μοντέλο εκπαιδεύτηκε σε 10 χρόνια δεδομένων 500+ μετοχών για να μάθει γενικά μοτίβα της αγοράς.")
            
            with col3:
                s_label = data['nlp_sentiment_label']
                s_score = data['nlp_sentiment_score']
                st.info(f" **Ειδήσεις (NLP - Live):**\n\nΤάση: **{str(s_label).strip()}**\n\n(Σκορ: {s_score})")
                # Τίτλοι ειδήσεων με το σκορ τους
                headlines = data.get('news_headlines', [])
                with st.expander(f"Τίτλοι ειδήσεων ({len(headlines)}) - πηγή: {data.get('news_source', '-')}"):
                    if not headlines:
                        st.write("Δεν βρέθηκαν ειδήσεις.")
                    for h in headlines:
                        st.write(f"`{h['score']:+.2f}` {h['title']}")
            
            st.divider()
            st.subheader("Τελική Συνδυαστική Πρόβλεψη")
            
            final_conf = data['final_confidence']
            final_dir = data['final_direction']
            conf_level = data['confidence_level']
            
            st.markdown(f"### {conf_level} {final_dir}")
            st.markdown(f"**Ποσοστό Σιγουριάς Συστήματος: {final_conf}%**")
            st.progress(final_conf / 100.0)
            effect = data.get('nlp_effect_points', 0.0)
            st.caption(f"Οι ειδήσεις μετακίνησαν την πιθανότητα ανόδου του Random Forest κατά {effect:+.1f} ποσοστιαίες μονάδες.")
            st.caption("Το τελικό ποσοστό προκύπτει από τη συνδυασμένη ένδειξη (log-odds) του γενικού S&P 500 μοντέλου και του live συναισθήματος ειδήσεων: όταν συμφωνούν, η σιγουριά αυξάνεται.")
            st.caption(f"Έκδοση μοντέλου (MLflow Registry): v{data.get('model_version', '-')}")
            t = data.get('timings_ms', {})
            if t:
                st.caption(f"Χρόνος απόκρισης: {t.get('total_ms', 0):.0f} ms (δεδομένα αγοράς {t.get('market_data_ms', 0):.0f} ms, "
                           f"μοντέλο {t.get('ml_inference_ms', 0):.0f} ms, ειδήσεις {t.get('news_sentiment_ms', 0):.0f} ms)")
                
        elif response.status_code == 404:
             st.error(f"❌ {response.json().get('detail')}")
        else:
            st.warning(f"Σφάλμα API: {response.json().get('detail', 'Άγνωστο σφάλμα')}")
            
    except requests.exceptions.Timeout:
        st.error("Το API δεν απάντησε εγκαίρως. Δοκιμάστε ξανά.")
    except requests.exceptions.ConnectionError:
        st.error("Αποτυχία Σύνδεσης! Το FastAPI δεν λειτουργεί.")