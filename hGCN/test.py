from utils import accuracy, hybrid_loss

def test(data, model):
    features, adj, idx_test, labels, label_mapping = data
    model.eval()
    output, encoding = model(features, adj)
    loss_test = hybrid_loss(output[idx_test], encoding[idx_test], labels[idx_test])

    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_k_test = top_k_accuracy(output[idx_test], labels[idx_test], 3)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "accuracy@3= {:.4f}".format(acc_k_test.item()))
    
    classes = label_mapping
    labels = labels.clone().cpu()
    output = output.clone().cpu()
    plot_confusion_matrix(labels[idx_test], output[idx_test], 
                          classes, normalize=True, filename='data/full_set_confusion')