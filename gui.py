from tkinter import *                       
from random import choice                       
                                    
ask   = ["hi", "hello"]                         
hi    = ["hi", "hello", "Hello too"]                    
error = ["sorry, i don't know", "what u said?" ]            




root = Tk()
root.geometry("700x700")  
root.resizable(0,0)                           
user = StringVar()                          
bot  = StringVar()                          
                                    
root.title(" VISION BOT ")                  
Label(root, text=" user : ").pack(side=LEFT)                
Entry(root, textvariable=user).pack(side=LEFT)          
Label(root, text=" Bot  : ").pack(side=LEFT)                
Entry(root, textvariable=bot).pack(side=LEFT)               
                            
                                
def main():                             
       question = user.get()                        
       if question in ask:                      
             bot.set(choice(hi))                    
       else:                                
             bot.set(choice(error))                 
                                
Button(root, text="speak", command=main).pack(side=LEFT)        
                                    
mainloop()