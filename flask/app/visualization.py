# Documento para funciones de visulización y algunas métricas.
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from inspect import signature
import os
from wordcloud import WordCloud

# Función para calcular precision, recall y f1_score
def prec_rec_f1(y_true, y_pred, average='weighted'):
    """Calcula las métricas precision, recall y f1_score

    Devuelve un array con los resultados de las métricas.

    Parámetros:
    y_true -- las etiquetas verdaderas de las instancias.
    y_pred -- las etiquetas calculadas por el modelo.
    average -- Indica cómo se calcularán estas medidas, puede ser 'micro', 'macro', 'weighted', 'binary'. Por defecto es 'weighted'
    """
    results = precision_recall_fscore_support(y_true, y_pred, average)
    return results

# FUnción para calcular la métrica 'precision'
def precision(y_true, y_pred):
    """Calcula la métrica precision

    Devuelve la métrica precision

    Parámetros:
    y_true -- las etiquetas verdaderas de las instancias.
    y_pred -- las etiquetas calculadas por el modelo.
    """
    prec = precision_score(y_true, y_pred)
    return prec

# Función para calcular la métrica 'recall'
def recall(y_true, y_pred):
    """Calcula la métrica recall

    Devuelve la métrica recall

    Parámetros:
    y_true -- las etiquetas verdaderas de las instancias.
    y_pred -- las etiquetas calculadas por el modelo.
    """
    rec = recall_score(y_true, y_pred)
    return rec

# Función para calcular la métrica F1_score:
def f1_sco(y_true, y_pred):
    """Calcula la métrica F1-Score

    Devuelve la métrica F1-Score

    Parámetros:
    y_true -- las etiquetas verdaderas de las instancias.
    y_pred -- las etiquetas calculadas por el modelo.
    """
    f1 = f1_score(y_true, y_pred)
    return f1

# Función para guardar una gráfica comparativa entre recall y precision
def recall_precision_plot(y_true, y_pred, name):
    print('ENTRO EN RECALL_PRECISION_PLOT')
    plt.figure()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, color='b', alpha=0.2, **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

    aux_path = os.getcwd()
    print(aux_path)
    path = aux_path + '/app/static/images/' + name + '.png'
    print('FIGURA AÚN NO GUARDADA EN: ' + path)
    plt.savefig(path)
    print('FIGURA GUARDADA EN: ' + path)


def generate_workcloud(text, name):
    plt.figure()
    t = " ".join(word for word in text)
    # wordcloud = WordCloud(background_color="none", mode='RGBA').generate(t)
    wordcloud = WordCloud(background_color="rgba(255, 221, 204, 1)", mode='RGBA').generate(t)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    aux_path = os.getcwd()
    print(aux_path)
    path = aux_path + '/app/static/images/' + str(name) + '.png'
    print(os.path)
    plt.savefig(path, transparent=True)
