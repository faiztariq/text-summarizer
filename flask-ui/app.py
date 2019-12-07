"""
Author : Faiz Tariq
Date : 11/21/2019
Desc : Abstractive Text Summarization using NLP - UI
"""

import os
from flask import Flask, render_template, flash, request
from wtforms import Form, TextAreaField, validators, SubmitField
from text_summarization.text_summary import predict_summary

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'M@ch!neLe@rn!ng'


class ReusableForm(Form):
    #Init form control
    review = TextAreaField('Review', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():

    """This method handles request"""

    form = ReusableForm(request.form)

    if request.method == 'POST':
        review = request.form['review']

    if form.validate():
        # Emit the comment.
        flash(predict_summary(review))
    else:
        flash('Error: Please provide with a review. ')

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run()
