from tkinter import *
from tkinter import filedialog, messagebox

def openFile():
    clearText()
    tf = filedialog.askopenfilename(
        initialdir="/",
        title="Open Text file",
        filetypes=(("Text Files", "*.txt"),)
    )
    pathh.insert(END, tf)
    tf = open(tf)
    data = tf.read()
    txtarea.insert(END, data)
    tf.close()


def clearText():
    txtarea.delete('1.0', END)
    pathh.delete(0, END)
    pathh.insert(0, "")


def showStats():
    novi = Toplevel()
    canvas = Canvas(novi, width=800, height=600)
    canvas.pack(expand=YES, fill=BOTH)
    stats = PhotoImage(file='./data/diferenta-bn-flip.png')
    canvas.create_image(80, 80, image=stats, anchor=NW)
    canvas.gif1 = stats


def runClassifier():
    messagebox.showinfo('Result', "The message is offensive!")
    messageToBeClassified = txtarea.get("1.0", "end")


ws = Tk()
ws.title("Offensive Language Detection for Romanian Language")
ws.geometry("800x600")
ws['bg'] = '#fb0'

label = Label(ws, text="Scrieti un mesaj pentru a fi clasificat", font="Helvetica 16 bold italic", fg="white",
              bg="#fb0")
label.pack()

txtarea = Text(ws, width=80, height=30)
txtarea.pack(pady=20)

pathh = Entry(ws)
pathh.pack(side=LEFT, expand=True, fill=X, padx=20)

Button(ws, text="Show Stats", command=showStats).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Clear", command=clearText).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Run", command=runClassifier).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Open File", command=openFile).pack(side=RIGHT, expand=True, fill=X, padx=20)

ws.mainloop()
