# -*- coding: utf-8 -*-

from flask import Flask, request, redirect
import numpy as np

from beholder import Beholder

app = Flask(__name__)


@app.route("/")
def root():
    def vector(vec):
        res = "<table><tr>"
        m, M = min(vec), max(vec)
        for i in res:
            col = 255.0 * (i-m)/(M-m)
            res += "<td style=background:rgb({0},{0},{0})></td>".format(int(col))
        res += "</tr></table>"
        return res

    query = request.args.get('query', '')
    text = "<style>body { padding: 2em;font-size: 200%; font-family: 'Droid Sans' } * { font: inherit }</style>"
    text += "Query: <form style=display:inline method=get><input id=a type=text value='%s' name=query /><input type=submit /></form><p>" % query
    if query:
        text += "<div style='margin:2em 0'>"
        results = se.search(query, viz=False, top=30)
        for res in results:
            text += "<div style='display:inline-block;width:300px;height:200px;margin:0 1em 1em 0'>"
            text += "<img src='%s' style=max-width:300px;max-height:200px>" % ("data:image/jpeg;base64,%s" % open(res).read().encode('base64').replace('\n', ''))
            text += "</div>"
        text += "</div>"

    text += u"<p style=color:#ccc>This is a prototype, don’t look for security bugs, pretty please"
    text += "<script>document.getElementById('a').focus()</script>"
    return text

if __name__ == "__main__":
    	se = Beholder()
	app.run(host='0.0.0.0', port=41113, debug=True)
    
