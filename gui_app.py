# gui_app.py

import tkinter as tk
from tkinter import font
import main  # Imports our main.py file
from tensorflow.keras.models import load_model
import pickle
import sys
from PIL import Image, ImageTk, ImageOps # <-- Import Pillow

# --- 1. Load Model and Tokenizer (Do this once at startup) ---
try:
    print("Loading Keras model 'sencnntive_model.h5'...")
    model = load_model('sencnntive_model.h5')
    print("Loading tokenizer 'tokenizer.pickle'...")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded successfully.")
except FileNotFoundError:
    print("\n" + "="*50)
    print("ERROR: Model or tokenizer files not found.")
    print("Please run 'python train_model.py' first to create these files.")
    print("="*50 + "\n")
    sys.exit() # Exit the app if files aren't found
except Exception as e:
    print(f"An error occurred while loading files: {e}")
    sys.exit()


# --- 2. Define GUI Colors and Fonts ("Techie" Theme) ---
BG_COLOR = "#2B2B2B"       # Dark grey background
TEXT_COLOR = "#E0E0E0"     # Light grey text
ENTRY_BG = "#3C3F41"       # Lighter grey for entry box
BUTTON_BG = "#0D638C"      # A strong tech-blue
BUTTON_FG = "#FFFFFF"      # White button text
RESULT_COLOR = "#26A69A"   # A teal/cyan for the result
CREDIT_COLOR = "#888888"   # Dim color for credit text
SECONDARY_BG = "#3C3F41"   # Background for report boxes

FONT_NORMAL = ("Consolas", 12)
FONT_BOLD = ("Consolas", 14, "bold")
FONT_RESULT = ("Consolas", 12, "bold")
FONT_CREDIT = ("Consolas", 9)
FONT_REPORT = ("Courier New", 11) # <-- Font for reports
FONT_TITLE = ("Consolas", 12, "bold")

# --- 3. Define GUI Functions ---
def analyze_post():
    """
    Gets text from the entry box, calls the prediction functions,
    and updates the result label.
    """
    url = url_entry.get()
    if not url:
        result_label.config(text="Please enter a Reddit URL first.")
        return

    # Clear old result and show loading
    result_label.config(text="Analyzing... Fetching post...", fg=TEXT_COLOR)
    root.update_idletasks() # Force GUI to update

    # 1. Get text from Reddit
    post_text = main.get_text_from_reddit(url)

    # Check for fetch error
    if post_text.startswith("Error:"):
        result_label.config(text=post_text, fg="#FF5252") # Red for error
        return
        
    result_label.config(text="Analyzing... Predicting sentiment...", fg=TEXT_COLOR)
    root.update_idletasks() # Force GUI to update

    # 2. Predict sentiment
    sentiment_result = main.predict_sentiment(post_text, model, tokenizer)
    
    # 3. Show final result
    if sentiment_result.startswith("Error:"):
        result_label.config(text=sentiment_result, fg="#FF5252") # Red for error
    else:
        result_label.config(text=sentiment_result, fg=RESULT_COLOR)

# =================================================================
# ⬇️⬇️ NEW FUNCTION TO SHOW EVALUATION ⬇️⬇️
# =================================================================
def show_evaluation_window():
    """
    Opens a new Toplevel window to display evaluation metrics.
    """
    try:
        # --- Create New Window ---
        eval_window = tk.Toplevel(root)
        eval_window.title("Model Evaluation Report")
        eval_window.configure(bg=BG_COLOR)
        eval_window.geometry("1000x800")

        # --- Load Images ---
        # We need to keep a reference to the images
        eval_window.img_accuracy = ImageTk.PhotoImage(
            Image.open("model_accuracy.png").resize((480, 240))
        )
        eval_window.img_loss = ImageTk.PhotoImage(
            Image.open("model_loss.png").resize((480, 240))
        )
        
        # --- Load Report Text ---
        with open("classification_report.txt", "r") as f:
            report_text = f.read()
        with open("confusion_matrix.txt", "r") as f:
            matrix_text = f.read()

        # --- Create Frames for Layout ---
        # Main frame for padding
        main_frame = tk.Frame(eval_window, bg=BG_COLOR, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top frame for plots (side-by-side)
        plot_frame = tk.Frame(main_frame, bg=BG_COLOR)
        plot_frame.pack(fill=tk.X, pady=10)

        # Bottom frame for reports (side-by-side)
        report_frame = tk.Frame(main_frame, bg=BG_COLOR)
        report_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # --- Display Plots ---
        plot_acc_frame = tk.Frame(plot_frame, bg=SECONDARY_BG, bd=1, relief=tk.SOLID)
        plot_acc_frame.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH)
        
        tk.Label(plot_acc_frame, text="Model Fit: Accuracy", font=FONT_TITLE, bg=SECONDARY_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.Label(plot_acc_frame, image=eval_window.img_accuracy, bg=SECONDARY_BG).pack(padx=10, pady=10)

        plot_loss_frame = tk.Frame(plot_frame, bg=SECONDARY_BG, bd=1, relief=tk.SOLID)
        plot_loss_frame.pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.BOTH)
        
        tk.Label(plot_loss_frame, text="Model Fit: Loss", font=FONT_TITLE, bg=SECONDARY_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.Label(plot_loss_frame, image=eval_window.img_loss, bg=SECONDARY_BG).pack(padx=10, pady=10)

        # --- Display Reports ---
        report_text_frame = tk.Frame(report_frame, bg=SECONDARY_BG, bd=1, relief=tk.SOLID)
        report_text_frame.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH)
        
        tk.Label(report_text_frame, text="Classification Report", font=FONT_TITLE, bg=SECONDARY_BG, fg=TEXT_COLOR).pack(pady=5)
        report_label = tk.Label(report_text_frame, text=report_text, font=FONT_REPORT, justify=tk.LEFT, bg=SECONDARY_BG, fg=TEXT_COLOR)
        report_label.pack(padx=20, pady=10, expand=True)
        
        matrix_text_frame = tk.Frame(report_frame, bg=SECONDARY_BG, bd=1, relief=tk.SOLID)
        matrix_text_frame.pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.BOTH)

        tk.Label(matrix_text_frame, text="Confusion Matrix", font=FONT_TITLE, bg=SECONDARY_BG, fg=TEXT_COLOR).pack(pady=5)
        matrix_label = tk.Label(matrix_text_frame, text=matrix_text, font=FONT_REPORT, justify=tk.LEFT, bg=SECONDARY_BG, fg=TEXT_COLOR)
        matrix_label.pack(padx=20, pady=10, expand=True)
        
    except FileNotFoundError:
        # Show an error in the main window if files are missing
        result_label.config(text="Error: Evaluation files not found. Run train_model.py first.", fg="#FF5252")
    except Exception as e:
        result_label.config(text=f"Error loading reports: {e}", fg="#FF5252")


# --- 4. Create the Main GUI Window ---
root = tk.Tk()
root.title("SenCNNtive - Sentiment Analyzer")
root.geometry("600x400") # Made window a little taller for new button
root.configure(bg=BG_COLOR)
root.resizable(False, False) # Lock window size

# Create a main frame for padding
main_frame = tk.Frame(root, bg=BG_COLOR, padx=20, pady=20)
main_frame.pack(expand=True, fill=tk.BOTH)

# --- Widgets ---
# Title Label
title_label = tk.Label(main_frame, 
                       text="SenCNNtive: Reddit Post Analyzer", 
                       font=FONT_BOLD, 
                       bg=BG_COLOR, 
                       fg=BUTTON_BG)
title_label.pack(pady=(0, 20))

# URL Entry
url_label = tk.Label(main_frame, 
                     text="Enter Reddit Post URL:", 
                     font=FONT_NORMAL, 
                     bg=BG_COLOR, 
                     fg=TEXT_COLOR)
url_label.pack(pady=5)

url_entry = tk.Entry(main_frame, 
                     font=FONT_NORMAL, 
                     width=60, 
                     bg=ENTRY_BG, 
                     fg=TEXT_COLOR, 
                     insertbackground=TEXT_COLOR, # Cursor color
                     relief=tk.FLAT,
                     borderwidth=5)
url_entry.pack(pady=5, ipady=5) # ipady makes it taller

# Analyze Button
analyze_button = tk.Button(main_frame, 
                           text="Analyze Sentiment", 
                           font=FONT_NORMAL, 
                           bg=BUTTON_BG, 
                           fg=BUTTON_FG,
                           activebackground="#117EB5", # Color when clicked
                           activeforeground=BUTTON_FG,
                           command=analyze_post,
                           relief=tk.FLAT,
                           cursor="hand2")
analyze_button.pack(pady=20, ipady=8, ipadx=10)

# =================================================================
# ⬇️⬇️ THIS IS THE FIXED LINE ⬇️⬇️
# =================================================================

# Result Label
result_label = tk.Label(main_frame, 
                        text="Awaiting analysis...", 
                        font=FONT_RESULT, 
                        bg=BG_COLOR, 
                        fg=TEXT_COLOR,
                        wraplength=550) # Wrap text if it's too long
result_label.pack(pady=10)


# =================================================================
# ⬇️⬇️ NEW EVALUATION BUTTON ⬇️⬇️
# =================================================================
eval_button = tk.Button(main_frame, 
                         text="Show Model Evaluation", 
                         font=FONT_NORMAL, 
                         bg=SECONDARY_BG, # A different color
                         fg=TEXT_COLOR,
                         activebackground="#4E5254",
                         activeforeground=TEXT_COLOR,
                         command=show_evaluation_window, # <-- Links to the new function
                         relief=tk.FLAT,
                         cursor="hand2")
eval_button.pack(pady=10, ipady=4, ipadx=5)


# --- Credit Label (at the very bottom) ---
credit_label = tk.Label(root, 
                        text="Made With \u2764 By Ankan", # \u2764 is the heart symbol
                        font=FONT_CREDIT,
                        bg=BG_COLOR,
                        fg=CREDIT_COLOR)
credit_label.pack(side="bottom", pady=5)


# --- 5. Start the Application ---
print("Starting GUI application...")
root.mainloop()