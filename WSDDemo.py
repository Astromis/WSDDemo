# all the imports
from __future__ import with_statement
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
from contextlib import closing
from wsd import wsd

    
# configuration
DEBUG = False

# create our little application :)
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

    
@app.route('/')
def show_entries():
    #cur = g.db.execute('select title, text from entries order by id desc')
    #entries = [dict(title=row[0], text=row[1]) for row in cur.fetchall()]
    return render_template('show_entries.html', synonyms='', best_defs='', defs={})


@app.route('/wsd', methods=['POST'])
def wsd_compute():
    #session['flag'] = 1
    synonyms, best_def, defs = wsd(request.form['passage'], request.form['quary'])
    return render_template('show_entries.html', synonyms=synonyms, best_def=best_def, defs=defs)

if __name__ == '__main__':

    app.run(debug=False)
