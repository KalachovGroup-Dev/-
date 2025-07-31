import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Настройки страницы
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка модели
@st.cache_resource
def load_model():
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    return model, features

def main():
    st.title("Диагностика диабета")
    model, feature_columns = load_model()

    # Разделение на две колонки
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.form("patient_form"):
            st.subheader("Демографические данные")
            gender = st.radio("Пол", ["Мужской", "Женский"])
            age = st.slider("Возраст", 1, 100, 45)

            st.subheader("Медицинские показатели")
            hypertension = st.checkbox("Гипертония")
            heart_disease = st.checkbox("Заболевания сердца")
            bmi = st.number_input("Индекс массы тела (ИМТ)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            hba1c = st.number_input("Уровень HbA1c (%)", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
            glucose = st.number_input("Уровень глюкозы (mg/dL)", min_value=70, max_value=300, value=100)

            st.subheader("Привычки")
            smoking = st.selectbox("Курение", [
                "Нет информации",
                "Никогда не курил(а)",
                "Бросил(а)",
                "Курю сейчас",
                "Не курю сейчас",
                "Курил(а) когда-либо"
            ])

            submitted = st.form_submit_button("Оценить риск")

    with col2:
        if submitted:
            with st.spinner('Анализируем данные...'):
                # Преобразование введенных данных
                gender_encoded = 1 if gender == "Мужской" else 0
                smoking_mapping = {
                    "Нет информации": "No Info",
                    "Никогда не курил(а)": "never",
                    "Бросил(а)": "former",
                    "Курю сейчас": "current",
                    "Не курю сейчас": "not current",
                    "Курил(а) когда-либо": "ever"
                }
                smoking_encoded = f"smoking_{smoking_mapping[smoking].replace(' ', '_')}"

                # Создаем DataFrame
                input_data = {
                    'gender': [gender_encoded],
                    'age': [age],
                    'hypertension': [int(hypertension)],
                    'heart_disease': [int(heart_disease)],
                    'bmi': [bmi],
                    'HbA1c_level': [hba1c],
                    'blood_glucose_level': [glucose]
                }

                # Добавляем smoking фичи
                for col in feature_columns:
                    if col.startswith('smoking'):
                        input_data[col] = [1 if col == smoking_encoded else 0]

                df = pd.DataFrame(input_data)[feature_columns]

                # Предсказание
                prediction = model.predict(df)
                proba = model.predict_proba(df)[0][1]

                # Визуализация результатов
                st.subheader("Результат:")

                # График вероятности
                fig1, ax1 = plt.subplots(figsize=(8, 2))
                ax1.barh(['Риск диабета'], [proba], color='#ff6b6b' if proba > 0.5 else '#51cf66')
                ax1.set_xlim(0, 1)
                ax1.set_title(f"Вероятность диабета: {proba * 100:.1f}%", pad=20)
                st.pyplot(fig1)

                # Вывод
                if proba > 0.7:
                    st.error("Высокий риск диабета - рекомендуется консультация врача!")
                elif proba > 0.3:
                    st.warning("Умеренный риск - рекомендуется профилактика")
                else:
                    st.success("Низкий риск")

                # Матрица важности признаков
                try:
                    st.subheader("Влияние параметров на прогноз")
                    importance = model.feature_importances_
                    feat_imp = pd.Series(importance, index=feature_columns).sort_values(ascending=True)

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    feat_imp.plot(kind='barh', ax=ax2, color='#5f6caf')
                    ax2.set_title("Важность признаков в модели")
                    st.pyplot(fig2)
                except:
                    st.info("Информация о важности признаков недоступна")
        else:
            st.info("Заполните форму и нажмите 'Оценить риск'")

    # Дополнительная информация в сайдбаре
    st.sidebar.header("О приложении")
    st.sidebar.markdown("""
    Это учебное приложение использует модель машинного обучения (XGBoost) 
    для оценки риска развития диабета на основе клинических параметров.
    """)

    st.sidebar.markdown("Используемые параметры:")
    st.sidebar.markdown("""
    - Демографические данные (возраст, пол)
    - Медицинские показатели (ИМТ, HbA1c, глюкоза)
    - История болезней (гипертония, болезни сердца)
    - Привычки (курение)
    """)

    st.sidebar.markdown("Метрики модели:")
    st.sidebar.markdown("""
    - Точность: 92%
    - Precision: 96% (для класса 'здоровый') и 85% (для класса 'больной')
    - Recall: 92% (для класса 'здоровый' и 'больной')
    - F1-score: 94% (для класса 'здоровый') и 88% (для класса 'больной')
    """)

if __name__ == '__main__':
    main()