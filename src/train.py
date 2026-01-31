import os
import joblib

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.temporal_generator import create_temporal_sequences
from src.models import build_lstm, build_xgboost
from src.config import *

df = load_raw_data(DATA_PATH)
X_train, X_test, y_train, y_test = preprocess_data(df)

X_seq, y_seq = create_temporal_sequences(X_train, y_train, SEQUENCE_LENGTH)

lstm_model = build_lstm((X_seq.shape[1], X_seq.shape[2]))
lstm_model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE)

os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
lstm_model.save(OUTPUT_MODEL_PATH + "lstm_model.h5")

xgb_model = build_xgboost()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, OUTPUT_MODEL_PATH + "xgboost_model.pkl")
