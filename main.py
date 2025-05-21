import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

class App:
    def __init__(self, master):
        self.master = master
        master.title("Modelos de ML: Regresión y Clasificación")
        master.geometry("800x600")

        self.label = tk.Label(master, text="📂 Cargar archivo CSV con columna 'retailvalue':")
        self.label.pack(pady=10)

        self.load_button = tk.Button(master, text="Cargar CSV", command=self.load_csv)
        self.load_button.pack()

        self.run_button = tk.Button(master, text="Ejecutar Modelos", command=self.run_models, state='disabled')
        self.run_button.pack(pady=10)

        self.result_box = tk.Text(master, height=30, width=100)
        self.result_box.pack(pady=10)

        self.df = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            if 'retailvalue' not in self.df.columns:
                messagebox.showerror("Error", "El archivo no contiene la columna 'retailvalue'")
                return

            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, f"✅ CSV cargado correctamente: {file_path}\n\n")
            self.result_box.insert(tk.END, f"🔍 Primeras filas:\n{self.df.head()}\n")
            self.run_button.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error al leer CSV", str(e))

    def run_models(self):
        df = self.df.copy()

        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)

        X = df.drop(columns='retailvalue')
        y = df['retailvalue']

        # Preprocesamiento
        X = X.fillna(X.mean(numeric_only=True))
        X_num = X.select_dtypes(include=np.number)

        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        output = "\n🎯 MODELOS DE REGRESIÓN (predicción de precios):\n"

        # Árbol de Decisión
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(X_train, y_train)
        pred_tree = tree.predict(X_test)
        output += f"\n🌳 Árbol de Decisión:\n  MAE: {mean_absolute_error(y_test, pred_tree):.2f}\n  R2 Score: {r2_score(y_test, pred_tree):.4f}"

        # KNN
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        pred_knn = knn.predict(X_test_scaled)
        output += f"\n👥 KNN:\n  MAE: {mean_absolute_error(y_test, pred_knn):.2f}\n  R2 Score: {r2_score(y_test, pred_knn):.4f}"

        # Clasificación binaria con Regresión Logística
        output += "\n\n CLASIFICACIÓN BINARIA (valor alto vs bajo):\n"
        median = y.median()
        y_binary = (y > median).astype(int)

        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_num, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
        X_train_cls_scaled = scaler.fit_transform(X_train_cls)
        X_test_cls_scaled = scaler.transform(X_test_cls)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_cls_scaled, y_train_cls)
        pred_cls = clf.predict(X_test_cls_scaled)

        acc = accuracy_score(y_test_cls, pred_cls)
        f1 = f1_score(y_test_cls, pred_cls)
        output += f"  Regresión Logística:\n  Accuracy: {acc:.4f}\n  F1 Score: {f1:.4f}"

        # Mostrar en GUI
        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, output)

# Ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
