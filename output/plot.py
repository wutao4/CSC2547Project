import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_LOSS_PATH = '%s/run-train-tag-Loss_Epoch_%s.csv'
TEST_LOSS_PATH = '%s/run-test-tag-Loss_Epoch_%s.csv'
METRIC_PATH = '%s/run-test-tag-Metric_%s.csv'


def plot_cleargrasp_loss(density):
    assert density in ['Dense', 'Sparse']

    cleargrasp_train = pd.read_csv(TRAIN_LOSS_PATH % ('cleargrasp', density))
    cleargrasp_test = pd.read_csv(TEST_LOSS_PATH % ('cleargrasp', density))
    norm_train = pd.read_csv(TRAIN_LOSS_PATH % ('cleargrasp-norm', density))
    norm_test = pd.read_csv(TEST_LOSS_PATH % ('cleargrasp-norm', density))

    plt.plot(cleargrasp_train['Step'], cleargrasp_train['Value'], 'b-', label='baseline-train')
    plt.plot(cleargrasp_test['Step'], cleargrasp_test['Value'], 'b--', label='baseline-test')
    plt.plot(norm_train['Step'], norm_train['Value'], 'r-', label='normalized-train')
    plt.plot(norm_test['Step'], norm_test['Value'], 'r--', label='normalized-test')

    plt.title('%s Point Cloud' % density)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_cleargrasp_metric(metric):
    assert metric in ['F-Score', 'ChamferDistance']

    cleargrasp = pd.read_csv(METRIC_PATH % ('cleargrasp', metric))
    norm = pd.read_csv(METRIC_PATH % ('cleargrasp-norm', metric))

    plt.plot(cleargrasp['Step'], cleargrasp['Value'], '', label='baseline')
    plt.plot(norm['Step'], norm['Value'], 'r', label='normalized')

    plt.title(metric)
    plt.xlabel('epoch')
    if metric == 'ChamferDistance':
        plt.ylim(0, 3)
    plt.legend()
    plt.show()


def plot_frankascan_loss(density):
    assert density in ['Dense', 'Sparse']

    fs_norm_train = pd.read_csv(TRAIN_LOSS_PATH % ('frankascan-norm', density))
    fs_norm_test = pd.read_csv(TEST_LOSS_PATH % ('frankascan-norm', density))
    fs_recenter_train = pd.read_csv(TRAIN_LOSS_PATH % ('frankascan-recenter', density))
    fs_recenter_test = pd.read_csv(TEST_LOSS_PATH % ('frankascan-recenter', density))
    fs_gtpcd_train = pd.read_csv(TRAIN_LOSS_PATH % ('frankascan-gtpcd', density))
    fs_gtpcd_test = pd.read_csv(TEST_LOSS_PATH % ('frankascan-gtpcd', density))

    plt.plot(fs_norm_train['Step'], fs_norm_train['Value'], 'b-', label='normalized-train')
    plt.plot(fs_norm_test['Step'], fs_norm_test['Value'], 'b--', label='normalized-test')
    plt.plot(fs_recenter_train['Step'], fs_recenter_train['Value'], 'r-', label='recentered-train')
    plt.plot(fs_recenter_test['Step'], fs_recenter_test['Value'], 'r--', label='recentered-test')
    plt.plot(fs_gtpcd_train['Step'], fs_gtpcd_train['Value'], 'g-', label='complete GT-train')
    plt.plot(fs_gtpcd_test['Step'], fs_gtpcd_test['Value'], 'g--', label='complete GT-test')

    plt.title('%s Point Cloud' % density)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_frankascan_metric(metric):
    assert metric in ['F-Score', 'ChamferDistance']

    norm = pd.read_csv(METRIC_PATH % ('frankascan-norm', metric))
    recenter = pd.read_csv(METRIC_PATH % ('frankascan-recenter', metric))
    gtpcd = pd.read_csv(METRIC_PATH % ('frankascan-gtpcd', metric))

    plt.plot(norm['Step'], norm['Value'], '', label='normalized')
    plt.plot(recenter['Step'], recenter['Value'], 'r', label='recentered')
    plt.plot(gtpcd['Step'], gtpcd['Value'], 'g', label='complete GT')

    plt.title(metric)
    plt.xlabel('epoch')
    if metric == 'ChamferDistance':
        plt.ylim(0, 25)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_cleargrasp_loss('Dense')
    plot_cleargrasp_loss('Sparse')
    plot_cleargrasp_metric('F-Score')
    plot_cleargrasp_metric('ChamferDistance')

    plot_frankascan_loss('Dense')
    plot_frankascan_loss('Sparse')
    plot_frankascan_metric('F-Score')
    plot_frankascan_metric('ChamferDistance')
