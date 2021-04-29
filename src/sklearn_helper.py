from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def confusion_Matrix(classifier, X_test, y_test):
    class_names = ['AMPs', 'NonAMPs']

    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                display_labels = class_names,
                                cmap=plt.cm.Blues, xticks_rotation='vertical')

    disp.ax_.set_title(" Confusion Matrix")

    print(disp.confusion_matrix)
    plt.grid(False)
    plt.show()

def create_roc_curve(labels, scores, positive_label):
  fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=positive_label)
  roc_auc = auc(fpr, tpr)

  plt.title('Receiver Operating Characteristic' )
  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.2])
  plt.ylim([-0.1,1.2])
  plt.xlabel('False Positive Rate or (1 - Specifity)')
  plt.ylabel('True Positive Rate or (Sensitivity)')
  plt.show()

def create_precision_recall_curve(model, y_test, x_test, pred):
    average_precision = average_precision_score(y_test, pred)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    disp = plot_precision_recall_curve(model, x_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')