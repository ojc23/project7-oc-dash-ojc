import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import shap
import pickle
import warnings
import streamlit as st
import streamlit.components.v1 as components
import requests, json
import ast

warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)
 
##__________________________________________
#Load Dataframe

st.sidebar.title("Filtre")

# limit nb customer to fater testing
rows = st.sidebar.slider("Selection nombre Max de clients ", 5, 100, 10)
#rows=50

x_test = pd.read_csv('./x_test.csv', nrows=rows).set_index('SK_ID_CURR')
y_test = pd.read_csv('./y_test.csv', nrows=rows).set_index('SK_ID_CURR')
x_train = pd.read_csv('./x_train.csv', nrows=rows).set_index('SK_ID_CURR')

# load the model from disk
filename = r'./model.pkl'
model = pickle.load(open(filename, 'rb'))

##__________________________________________
# calculate prediction probability
probas = model.predict_proba(x_test)[:,1]

##__________________________________________
# enrich the dataset with wistuler categorical infos
df_customer = pd.read_csv('./df_customer.csv').set_index('SK_ID_CURR')

##__________________________________________
## Start repoting

st.sidebar.subheader(f'Les informations du client')
st.write('''# Scoring App: Approbation d'un prêt
Cette Application permet de caluler le score de chaque cloent''')


def customer_stat():
	id_client = st.sidebar.selectbox('Select ID Client :', x_test.index.tolist())
	customer = id_client
	index = x_test.index.tolist().index(id_client)
	probas = model.predict_proba(x_test)[index,1]
	score = model.predict_proba(x_test)[index,0]
	predict = model.predict(x_test)[index]
	return customer, probas, predict, score, index

df_0=customer_stat()


##__________________________________________
# get the index of the customer
index=df_0[4]


##__________________________________________
def feature_importance():
	import plotly.express as px 
	#df=px.data.tips()
	Importance_df = pd.DataFrame((model.feature_importances_ / sum(model.feature_importances_))*100, index=x_test.columns, columns=['Features Importance'])
	Importance_df = Importance_df.sort_values(by = 'Features Importance')[-10::]
	fig=px.bar(Importance_df, orientation='h')
	plt.title('Features Importance')
	return(fig)
	
df_f=feature_importance()


##__________________________________________
# get information from API
def prediction(id_custo):
    url ='https://project7-api-ojc.herokuapp.com'
    headers = {'Content-Type': 'application/json'}
    headers = {
    'Accept': 'application/json',
    'Authorization': 'key ttn-account-notsharinganything',
        }
        
    params = {
        'running': True,
        'id_custo': id_custo,
    }

    response = requests.get(url, headers=headers, params=params)
    proba = float(ast.literal_eval(response.text)["proba_computed"])

    return(round(proba*100,1))
    
proba = prediction(index)


##__________________________________________

st.write(f'Le client selectionné est le :', df_0[0])

link = '[API](https://project7-api-ojc.herokuapp.com/?id_custo='+str(index)+')'
st.write(f'La probabilite de remboursement est de :', round(df_0[1]*100,1),'% ', 
		' check from (API) ', proba,'%',  )
st.markdown(link, unsafe_allow_html=True)
st.write(f'Peut on accorder un pret :', 'Oui' if df_0[2]==1 else 'Non')


##__________________________________________
st.write('''### Decision Explanation ''')

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
 

def shap_summary_4():
	explainer = shap.TreeExplainer(model, 
	                               model_output = 'probability',
	                               data = x_train,
	                               re_perturbation = 'interventional')

	shap_values = explainer.shap_values(y_test)

	i=index
	st_shap(shap.force_plot(explainer.expected_value, shap_values[i], x_test.iloc[i]))
	
	st.write('''In the plot above, the bold value is the model’s score 
	for this observation. Higher scores lead the model to predict 
	1 and lower scores lead the model to predict 0. The features 
	that were important to making the prediction for this observation 
	are shown in red and blue, with red representing features that 
	pushed the model score higher, and blue representing features 
	that pushed the score lower. Features that had more of an impact 
	on the score are located closer to the dividing boundary 
	between red and blue, and the size of that impact is represented 
	by the size of the bar.''')

	explainer = shap.Explainer(model, x_train)
	shap_values = explainer(y_test)
	shap.plots.bar(shap_values[i], max_display=12)
	st.pyplot()

shap_summary_4()

##__________________________________________

st.write('''### Customer information''')
st.write(df_customer)

st.sidebar.text(df_customer.iloc[index,:])

##__________________________________________

st.write('''### Feature importance globale
Display the globla Feature Importance oof the dataset''')
#
st.write(df_f)

st.write('''### Feature importance : impact of variables taking into account 
SHAP measures the impact of variables taking into account 
the interaction with other variables.
Shapley values calculate the importance of a feature by 
comparing what a model predicts with and without the feature. 
However, since the order in which a model sees features can 
affect its predictions, this is done in every possible order, 
so that the features are fairly compared.''')
def shap_summary():
	shap.initjs()
	filename = r'./shap_values_dash.pkl'
	shap_values = pickle.load(open(filename, 'rb'))

	#summary_plot
	shap.summary_plot(shap_values[:rows], x_test)
	st.pyplot()
shap_summary() 






