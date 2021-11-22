from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd


def confusion_mat(labeled):
    cm1 = confusion_matrix(labeled['label'], labeled['predict'])
    total1 = sum(sum(cm1))

    accuracy = (cm1[0, 0]+cm1[1, 1])/total1
    sensitivity = cm1[0, 0]/(cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1]/(cm1[1, 0] + cm1[1, 1])
    f1 = f1_score(labeled['label'], labeled['predict'])

    return {'accuracy': [accuracy],
            'sensitivity': [sensitivity],
            'specificity': [specificity],
            'f1': [f1]}


def predict_data(data, cutoff):

    data['predict'] = data['sim'] > cutoff
    print("there are %i total matches", len(data))
    print("there are %i actual matches", data['label'].sum())
    print("predicted %i relevant matches", data['predict'].sum())

    return data


if __name__ == '__main__':

    labeled = pd.read_csv('../labeled.csv')
    labeled = labeled.fillna('')

    sim = []
    for i, row in labeled[['news', 'wiki']].iterrows():
        # embeddings = embed([row['news'], row['wiki']])
        # sim.append(spatial.distance.cosine(embeddings[0], embeddings[1]))
        sim.append(get_sim([row['news'], row['wiki']]))

    labeled['sim'] = sim

    # for cutoff in [0.6, 0.7, 0.75, 0.8, 0.85]:

    predicted = predict_data(labeled, 0.75)
    print(confusion_mat(predicted))
