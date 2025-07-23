import pickle
import os
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
from PIL import Image, ImageDraw
import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class DrawingClassifier:
    def __init__(self):
        self.class1 = self.class2 = self.class3 = None
        self.class1_counter = self.class2_counter = self.class3_counter = 1
        self.clf = LinearSVC(max_iter=10000)  # Increased max_iter for better convergence
        self.scaler = StandardScaler()
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None
        self.brush_width = 15
        self.last_x = None
        self.last_y = None

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Enter your project name:", parent=msg)
        if not self.proj_name:
            raise ValueError("Project name cannot be empty")

        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.load_from_data(data)
        else:
            self.class1 = simpledialog.askstring("Class 1", "First class name:", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "Second class name:", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "Third class name:", parent=msg)

            os.makedirs(self.proj_name)
            os.makedirs(f"{self.proj_name}/{self.class1}")
            os.makedirs(f"{self.proj_name}/{self.class2}")
            os.makedirs(f"{self.proj_name}/{self.class3}")

    def load_from_data(self, data):
        self.class1 = data['c1']
        self.class2 = data['c2']
        self.class3 = data['c3']
        self.class1_counter = data['c1c']
        self.class2_counter = data['c2c']
        self.class3_counter = data['c3c']
        self.clf = data['clf']
        self.proj_name = data['pname']

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"Drawing Classifier - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_brush)

        self.image1 = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.image1)

        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        # Buttons setup...
        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.grid(row=4, columnspan=3)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=self.brush_width, fill='black', capstyle=ROUND, smooth=TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill='black', width=self.brush_width)
        self.last_x = event.x
        self.last_y = event.y

    def reset_brush(self, event):
        self.last_x = None
        self.last_y = None

    def save(self, class_num):
        try:
            self.image1.save("temp.png")
            img = Image.open("temp.png")
            img.thumbnail((50, 50), Image.LANCZOS)

            class_dir = getattr(self, f"class{class_num}")
            counter = getattr(self, f"class{class_num}_counter")
            img.save(f"{self.proj_name}/{class_dir}/{counter}.png", "PNG")
            setattr(self, f"class{class_num}_counter", counter + 1)

            self.clear()
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def train_model(self):
        try:
            img_list = []
            class_list = []

            for class_num in [1, 2, 3]:
                class_dir = getattr(self, f"class{class_num}")
                counter = getattr(self, f"class{class_num}_counter")

                for i in range(1, counter):
                    img_path = f"{self.proj_name}/{class_dir}/{i}.png"
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_list.append(img.flatten())
                        class_list.append(class_num)

            if not img_list:
                raise ValueError("No training images found")

            X = np.array(img_list)
            y = np.array(class_list)

            # Normalize the data
            X = self.scaler.fit_transform(X)

            self.clf.fit(X, y)
            tkinter.messagebox.showinfo("Success", "Model trained successfully!")

        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Training failed: {str(e)}")

    def predict(self):
        try:
            if not hasattr(self.clf, 'classes_'):
                raise ValueError("Model not trained. Train first.")

            self.image1.save("temp.png")
            img = Image.open("temp.png")
            img.thumbnail((50, 50), Image.LANCZOS)
            img.save("predict.png", "PNG")

            predict_img = cv.imread("predict.png", cv.IMREAD_GRAYSCALE)
            if predict_img is None:
                raise ValueError("Failed to process image")

            predict_img = self.scaler.transform([predict_img.flatten()])
            prediction = self.clf.predict(predict_img)[0]

            class_name = getattr(self, f"class{prediction}")
            tkinter.messagebox.showinfo("Prediction", f"Predicted: {class_name}")

        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def rotate_model(self):
        models = [
            LinearSVC(max_iter=10000),
            KNeighborsClassifier(),
            LogisticRegression(max_iter=1000),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GaussianNB()
        ]

        current_type = type(self.clf)
        next_index = (models.index(next(m for m in models if isinstance(self.clf, type(m)))) + 1) % len(models)
        self.clf = models[next_index]

        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")
        tkinter.messagebox.showinfo("Model Changed", f"Switched to {type(self.clf).__name__}")

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 500, 500], fill="white")
        self.reset_brush(None)

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pickle")
        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump(self.clf, f)
            tkinter.messagebox.showinfo("Success", "Model saved!")

    def load_model(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as f:
                self.clf = pickle.load(f)
            tkinter.messagebox.showinfo("Success", "Model loaded!")
            self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_everything(self):
        data = {
            "c1": self.class1,
            "c2": self.class2,
            "c3": self.class3,
            "c1c": self.class1_counter,
            "c2c": self.class2_counter,
            "c3c": self.class3_counter,
            "clf": self.clf,
            "pname": self.proj_name
        }

        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)

        tkinter.messagebox.showinfo("Success", "Everything saved!")

    def on_closing(self):
        if tkinter.messagebox.askyesno("Quit", "Save before quitting?"):
            self.save_everything()
        self.root.destroy()


if __name__ == "__main__":
    DrawingClassifier()
