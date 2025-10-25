# sentiment_analysis.py

import tkinter as tk
from tkinter import messagebox
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------
# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

# -----------------------------
# Sentiment functions
def analyze_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(polarity, 3)

def analyze_vader(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(compound, 3)

# -----------------------------
# GUI Class
class SentimentGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sentiment Analysis - Enhanced Visualization")
        self.root.geometry("700x520")
        self.root.configure(bg="#e6f0fa")  # soft blue background

        # Header
        tk.Label(
            self.root, 
            text="💬 Sentiment Analysis Dashboard", 
            font=("Arial", 18, "bold"), 
            bg="#4a90e2", 
            fg="white", 
            pady=12
        ).pack(fill="x")

        # Input field
        frame_input = tk.Frame(self.root, bg="#e6f0fa")
        frame_input.pack(pady=15)
        tk.Label(frame_input, text="Enter a sentence:", font=("Arial", 16, "bold"), bg="#6b92ba").pack()
        self.text_entry = tk.Entry(frame_input, width=70, font=("Arial", 12))
        self.text_entry.pack(pady=5)

        # Analyze button (TEAL color)
        self.analyze_btn = tk.Button(
            self.root, text="Analyze Sentiment", font=("Arial", 12, "bold"),
            bg="#00897B", fg="black", activebackground="#26A69A",
            activeforeground="black", relief="raised", bd=3,
            command=self.analyze
        )
        self.analyze_btn.pack(pady=10)

        # Add hover effect
        self.analyze_btn.bind("<Enter>", lambda e: self.analyze_btn.config(bg="#26A69A"))
        self.analyze_btn.bind("<Leave>", lambda e: self.analyze_btn.config(bg="#00897B"))

        # Result area
        self.result_text = tk.StringVar()
        self.result_label = tk.Label(
            self.root, textvariable=self.result_text, font=("Arial", 14),
            bg="#e6f0fa", justify="left"
        )
        self.result_label.pack(pady=10)

        # Canvas for chart
        self.canvas = tk.Canvas(self.root, width=550, height=260, bg="#f8fbff", highlightbackground="#b0c4de")
        self.canvas.pack(pady=10)

        # Exit button
        tk.Button(
            self.root, text="Exit", font=("Arial", 12, "bold"),
            bg="#E53935", fg="white", command=self.root.destroy
        ).pack(pady=10)

    def sentiment_color(self, sentiment):
        """Return color for sentiment"""
        return {
            "Positive": "#4CAF50",   # Green
            "Negative": "#E53935",   # Red
            "Neutral": "#FFA726"     # Orange
        }.get(sentiment, "gray")

    def analyze(self):
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter a sentence.")
            return

        tb_sent, tb_score = analyze_textblob(text)
        vd_sent, vd_score = analyze_vader(text)

        self.result_text.set(f"TextBlob: {tb_sent} ({tb_score})\nVADER:    {vd_sent} ({vd_score})")
        self.result_label.config(fg=self.sentiment_color(tb_sent))

        # Draw colorful bar chart
        self.canvas.delete("all")
        max_height = 180
        bars = [tb_score, vd_score]
        labels = ["TextBlob", "VADER"]
        colors = [self.sentiment_color(tb_sent), self.sentiment_color(vd_sent)]

        for i, score in enumerate(bars):
            bar_height = (score + 1) / 2 * max_height  # scale from -1..1 to 0..max
            x0 = 130 + i * 200
            y0 = max_height - bar_height + 50
            x1 = x0 + 70
            y1 = max_height + 50

            # Bar rectangle
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=colors[i], outline="black")

            # Label under bar
            self.canvas.create_text(x0 + 35, y1 + 25, text=labels[i], font=("Arial", 12, "bold"))

            # Show numeric score beside bar
            percent_text = f"{round(score * 100)}%"
            self.canvas.create_text(x1 + 30, (y0 + y1) / 2, text=percent_text, font=("Arial", 11, "bold"), fill="#333")

    def run(self):
        self.root.mainloop()

# -----------------------------
if __name__ == "__main__":
    SentimentGUI().run()
