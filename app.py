from flask import Flask, render_template
import subprocess
app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def index():

    return render_template('index.html')

@app.route('/execute')


def execute():

    subprocess.run(['python', 'scratch.py'])

    return 'Python code executed successfully!'

if __name__ == '__main__':

    app.run(debug=True)




