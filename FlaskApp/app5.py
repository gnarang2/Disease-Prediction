import cv2
import numpy as np
import pickle
import dash_core_components as dcc
import dash_html_components as html
import dash
from flask import Flask
import os
import base64
from urllib.parse import quote as urlquote
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from dash.dependencies import Input, Output, State
import math

json_file = open('multiclass_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
multiclass_model = tf.keras.models.model_from_json(loaded_model_json)
multiclass_model.load_weights("multiclass_model.h5")
multiclass_model.compile(
      loss='categorical_crossentropy', 
      optimizer='adam', 
      metrics=['accuracy']
)

# json_file = open('corona_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# corona_model = tf.keras.models.model_from_json(loaded_model_json)
# corona_model.load_weights("corona_model.h5")
# corona_model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

# TODO
corona_model = tf.keras.models.load_model('covid_chest_xception.h5')
# compile model(?)
# Change model in line 69

n_clicks_prev = None

UPLOAD_DIRECTORY = "uploads/"

server = Flask(__name__)

app = dash.Dash(name='BondScoring_docker_app',
                server=server)

#app.config.supress_callback_exceptions = True
prefix = ''

if os.getenv('DASH_APP_NAME'):
    prefix = '/{0}/'.format(os.getenv('DASH_APP_NAME'))

app.config.update({
 	# remove the default of '/'
 	'routes_pathname_prefix': prefix,
 	# remove the default of '/'
 	'requests_pathname_prefix': prefix,
})

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def get_corona_model():
    return corona_model

def get_multiclass_model():
    return multiclass_model

def get_image():
    image = cv2.imread('uploads/img.jpg')
    image  = cv2.resize(image, (224, 224))
    image = np.resize(image,(1, 224, 224, 3))
    return image

def apicall():
    img = get_image()
    diseases_list1 = ["Not Corona", "Coronavirus"]
    diseases_list2 = ['Pneumothorax', 'Nodule', 'Consolidation', 'Mass', 'Cardiomegaly', 'Atelectasis', 'No Finding', 'Effusion', 'Infiltration', 'Pleural_Thickening']
    loaded_model = get_corona_model()
    result1 = loaded_model.predict(img/255)
    if(int(np.argmax(result1)) == 1):
        return diseases_list1[np.argmax(result1)] + " with probability {}%".format(round(100*result1[0][np.argmax(result1)], 2))
    loaded_model = get_multiclass_model()
    result2 = loaded_model.predict(img)
    return diseases_list2[np.argmax(result2)] + " and corona isn't present with probability {}%".format(round(100*result1[0][np.argmax(result1)], 2))

textBoxStyle = {"height": "auto", "margin-bottom": "auto",'fontWeight':'bold','fontSize':'0.9rem','text-align':'auto'}
logoStyle = {'height': '10%','width': '10%','margin':'auto','display':'inline-block'}
headingStyle = {'textAlign': 'center', 'color':'rgb(214,0,42)', 'margin':'auto' }
    
app.layout = html.Div(
    [
        html.H1('Disease Detection using Xrays',style=headingStyle),html.Br(),html.Br(),html.Br(),
    html.Div(
    [
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.Ul(id="file-list"),
    ],
    style={"max-width": "500px"},
),
	# html.Div(html.Img(src = 'https://www.spglobal.com/_assets/images/marketintelligence/logo-mi.png', style=logoStyle)),
	    # html.Div([
	    #         html.Div([html.P('Days To Maturity', style=textBoxStyle),
	    #                   dcc.Input(id="DaysToMaturity", type="number", value='806.0', ), ]),
	    #         html.Div([html.P('Days From Offer', style=textBoxStyle),
	    #                   dcc.Input(id="DaysFromOffer", type="number", value='2879.0', ), ]),
	    #         html.Div([html.P('Price', style=textBoxStyle),
	    #                   dcc.Input(id="Price", type="number", value='105.033', ), ]),
	    #         html.Div([html.P('Yield To Worst', style=textBoxStyle),
	    #                   dcc.Input(id="Yild2Worst", type="number", value='3.414', ), ]),
	    #         html.Div([html.P('Coupon Rate', style=textBoxStyle),
	    #                   dcc.Input(id="CouponRate", type="number", value='5.8', ), ]),
	    #         html.Div([html.P('Offering Amount', style=textBoxStyle),
	    #                   dcc.Input(id="OfferinAmnt", type="number", value='450.0', ), ]),
	    #         html.Div([html.P('Amount Outstanding', style=textBoxStyle),
	    #                   dcc.Input(id="AmntOut", type="number", value='450.0', ), ]),
	    #         html.Div([html.P('Offering Price', style=textBoxStyle),
	    #                   dcc.Input(id="OffPrice", type="number", value='99.8', ), ]),
	    #         html.Div([html.P('Offering Yield', style=textBoxStyle),
	    #                   dcc.Input(id="OffYild", type="number", value='5.832', ), ]),
	    #         html.Div([html.P('Principal Amount', style=textBoxStyle),
	    #                   dcc.Input(id="PrincAmnt", type="number", value='1000.0', ), ]),
	    #         html.Div([html.P('Duration', style=textBoxStyle),
	    #                   dcc.Input(id="Duration", type="number", value='2.078', ), ]),
	    #         html.Div([html.P('Convexity', style=textBoxStyle),
	    #                   dcc.Input(id="Convexity", type="number", value='0.03', ), ]),
	    #     ], style=dict(display='flex', flexWrap='wrap', width=400)),
	        # html.Br(),
	        html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
	                              id='button'), ]),
	        html.Br(),
	        html.Div(id="prediction-out", style={'color': 'black', 'font-style': 'italic', 'font-weight': 'bold'}),
			# html.Footer('Powered by Garvit Narang and Krishan Makkar',style={'textAlign': 'center', 'color':'black', 'margin':'150px' })
    ]
)

@app.callback(
    [Output("prediction-out", "children")],
    [Input("upload-data", "filename"), Input("upload-data", "contents"), Input('button', 'n_clicks')],
    # [State("DaysToMaturity", "value"), State("DaysFromOffer", "value"),State("Price", "value"), State("Yild2Worst", "value"),
    #  State("CouponRate", "value"), State("OfferinAmnt", "value"),State("AmntOut", "value"), State("OffPrice", "value"),
    #  State("OffYild", "value"), State("PrincAmnt", "value"),State("Duration", "value"), State("Convexity", "value")]
)

# def myFun(n_clicks, *argv):
#     # inputlst = []
#     # for arg in argv:
#     #     inputlst.append(arg)
#     # inputArray = np.asarray(inputlst)

#     if n_clicks is None :
#         return ' '
#     else:
#         return 'HI'
#         # return 'The Predicted Bond Score is : "{}"'.format(apicall(inputArray), "s")
    
def myfunc(uploaded_filenames, uploaded_file_contents, n_clicks):
    # """Save uploaded files and regenerate the file list."""
    global n_clicks_prev
    if(n_clicks != n_clicks_prev):
        n_clicks_prev = n_clicks
        val = apicall()
        return ('The Predicted Class is : {}'.format(val),)

    if uploaded_file_contents is not None and uploaded_filenames is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file("img.jpg", data)

    return (" ",)


if __name__ == '__main__':
    app.run_server(debug=True)