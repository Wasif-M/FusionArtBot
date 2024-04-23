from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)
@app.route('/')
def index():
    
    return render_template('index.html')


@app.route('/login')
def login():
    
    return render_template('login.html',method="GET")

@app.route("/signup")
def signup():
    return render_template("signup.html",method="GET")
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)