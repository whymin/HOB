#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle,os
from rdkit import Chem
from rdkit.Chem import Draw
import re
import codecs
import sys
from sklearn.decomposition import PCA
import joblib
from matplotlib.path import Path


save_index1=['ABC', 'ABCGG', 'nAcid', 'nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A', 'nAromAtom', 'nAromBond', 'nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nHetero', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s', 'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s', 'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se', 'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS1dv', 'AATS2dv', 'AATS3dv', 'AATS4dv', 'AATS5dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', 'AATS4d', 'AATS5d', 'AATS0s', 'AATS1s', 'AATS2s', 'AATS3s', 'AATS4s', 'AATS5s', 'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS4Z', 'AATS5Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS4m', 'AATS5m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v', 'AATS4v', 'AATS5v', 'AATS0se', 'AATS1se', 'AATS2se', 'AATS3se', 'AATS4se', 'AATS5se', 'AATS0pe', 'AATS1pe', 'AATS2pe', 'AATS3pe', 'AATS4pe', 'AATS5pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are', 'AATS4are', 'AATS5are', 'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS4p', 'AATS5p', 'AATS0i', 'AATS1i', 'AATS2i', 'AATS3i', 'AATS4i', 'AATS5i', 'ATSC0c', 'ATSC1c', 'ATSC2c', 'ATSC3c', 'ATSC4c', 'ATSC5c', 'ATSC6c', 'ATSC7c', 'ATSC8c', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d', 'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s', 'ATSC1s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', 'ATSC8s', 'ATSC0Z', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC0se', 'ATSC1se', 'ATSC2se', 'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC0pe', 'ATSC1pe', 'ATSC2pe', 'ATSC3pe', 'ATSC4pe', 'ATSC5pe', 'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are', 'ATSC3are', 'ATSC4are', 'ATSC5are', 'ATSC6are', 'ATSC7are', 'ATSC8are', 'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p', 'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0c', 'AATSC1c', 'AATSC2c', 'AATSC3c', 'AATSC4c', 'AATSC5c', 'AATSC0dv', 'AATSC1dv', 'AATSC2dv', 'AATSC3dv', 'AATSC4dv', 'AATSC5dv', 'AATSC0d', 'AATSC1d', 'AATSC2d', 'AATSC3d', 'AATSC4d', 'AATSC5d', 'AATSC0s', 'AATSC1s', 'AATSC2s', 'AATSC3s', 'AATSC4s', 'AATSC5s', 'AATSC0Z', 'AATSC1Z', 'AATSC2Z', 'AATSC3Z', 'AATSC4Z', 'AATSC5Z', 'AATSC0m', 'AATSC1m', 'AATSC2m', 'AATSC3m', 'AATSC4m', 'AATSC5m', 'AATSC0v', 'AATSC1v', 'AATSC2v', 'AATSC3v', 'AATSC4v', 'AATSC5v', 'AATSC0se', 'AATSC1se', 'AATSC2se', 'AATSC3se', 'AATSC4se', 'AATSC5se', 'AATSC0pe', 'AATSC1pe', 'AATSC2pe', 'AATSC3pe', 'AATSC4pe', 'AATSC5pe', 'AATSC0are', 'AATSC1are', 'AATSC2are', 'AATSC3are', 'AATSC4are', 'AATSC5are', 'AATSC0p', 'AATSC1p', 'AATSC2p', 'AATSC3p', 'AATSC4p', 'AATSC5p', 'AATSC0i', 'AATSC1i', 'AATSC2i', 'AATSC3i', 'AATSC4i', 'AATSC5i', 'MATS1c', 'MATS2c', 'MATS3c', 'MATS4c', 'MATS5c', 'MATS1dv', 'MATS2dv', 'MATS3dv', 'MATS4dv', 'MATS5dv', 'MATS1d', 'MATS2d', 'MATS3d', 'MATS4d', 'MATS5d', 'MATS1s', 'MATS2s', 'MATS3s', 'MATS4s', 'MATS5s', 'MATS1Z', 'MATS2Z', 'MATS3Z', 'MATS4Z', 'MATS5Z', 'MATS1m', 'MATS2m', 'MATS3m', 'MATS4m', 'MATS5m', 'MATS1v', 'MATS2v', 'MATS3v', 'MATS4v', 'MATS5v', 'MATS1se', 'MATS2se', 'MATS3se', 'MATS4se', 'MATS5se', 'MATS1pe', 'MATS2pe', 'MATS3pe', 'MATS4pe', 'MATS5pe', 'MATS1are', 'MATS2are', 'MATS3are', 'MATS4are', 'MATS5are', 'MATS1p', 'MATS2p', 'MATS3p', 'MATS4p', 'MATS5p', 'MATS1i', 'MATS2i', 'MATS3i', 'MATS4i', 'MATS5i', 'GATS1c', 'GATS2c', 'GATS3c', 'GATS4c', 'GATS5c', 'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS4dv', 'GATS5dv', 'GATS1d', 'GATS2d', 'GATS3d', 'GATS4d', 'GATS5d', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS4s', 'GATS5s', 'GATS1Z', 'GATS2Z', 'GATS3Z', 'GATS4Z', 'GATS5Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS4m', 'GATS5m', 'GATS1v', 'GATS2v', 'GATS3v', 'GATS4v', 'GATS5v', 'GATS1se', 'GATS2se', 'GATS3se', 'GATS4se', 'GATS5se', 'GATS1pe', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS5pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS5are', 'GATS1p', 'GATS2p', 'GATS3p', 'GATS4p', 'GATS5p', 'GATS1i', 'GATS2i', 'GATS3i', 'GATS4i', 'GATS5i', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1h', 'BCUTse-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'BCUTi-1h', 'BCUTi-1l', 'BalabanJ', 'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', 'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'RNCG', 'RPCG', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio', 'FCSP3', 'Xch-3d', 'Xch-4d', 'Xch-5d', 'Xch-6d', 'Xch-7d', 'Xch-3dv', 'Xch-4dv', 'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-6d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'AXp-3d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', 'AXp-2dv', 'AXp-3dv', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi', 'SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', 'NsLi', 'NssBe', 'NssssBe', 'NssBH', 'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2', 'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe', 'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', 'NsSeH', 'NdSe', 'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2', 'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', 'NssssPb', 'SsLi', 'SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', 'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP', 'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe', 'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'ECIndex', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', 'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', 'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_3', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B', 'fragCpx', 'fMF', 'nHBAcc', 'nHBDon', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'Kier1', 'Kier2', 'Kier3', 'Lipinski', 'GhoseFilter', 'FilterItLogS', 'VMcGowan', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MID_N', 'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'apol', 'bpol', 'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', 'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing', 'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing', 'n10aRing', 'n11aRing', 'n12aRing', 'nG12aRing', 'naHRing', 'n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing', 'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', 'nG12aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing', 'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n4FRing', 'n5FRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing', 'n9FaRing', 'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n11FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', 'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing', 'nRot', 'RotRatio', 'SLogP', 'SMR', 'TopoPSA(NO)', 'TopoPSA', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'VAdjMat', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2']
zero1=[728, 729, 730, 731, 733, 741, 747, 766, 767, 768, 769, 770, 771, 772, 774, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 798, 799, 800, 801, 803, 804, 805, 806, 807, 808, 809, 810, 812, 820, 826, 845, 846, 847, 848, 849, 850, 851, 853, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 877, 878, 879, 880, 882, 883, 884, 885, 1006, 1016, 1081, 1082, 1083, 1093, 1094, 1095, 1098, 1099, 1103, 1104, 1105, 1106, 1107, 1108, 1110, 1111, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1129, 1130, 1131, 1141, 1142, 1143, 1146, 1147, 1157, 1158, 1168, 1169, 1170, 1171, 1172, 1175, 1176, 1179, 1180, 1181, 1182, 1183, 1186, 1187, 1190, 1191, 1201, 1202]
avgs=np.array([ 6.34545605e+01,  4.65000052e+01,  4.00000000e+00,  8.00000000e+00,
        1.03815778e+02,  2.81450664e+00,  5.51413840e+00,  1.03815778e+02,
        1.40398171e+00,  5.31952082e+00,  7.53450818e+00,  4.26197263e-01,
        4.11135762e+00,  3.42296168e+05,  6.00519593e+03,  1.44838978e+01,
        2.90000000e+01,  3.50000000e+01,  1.79000000e+02,  8.20000000e+01,
        2.00000000e+00,  6.00000000e+00,  2.90000000e+01,  9.80000000e+01,
        1.00000000e+00,  5.80000000e+01,  1.60000000e+01,  2.10000000e+01,
        4.00000000e+00,  2.00000000e+00,  7.00000000e+00,  5.00000000e+00,
        2.00000000e+00,  4.00000000e+00,  7.00000000e+00,  1.19000000e+03,
        1.10400000e+03,  1.61800000e+03,  1.97400000e+03,  2.24500000e+03,
        2.71300000e+03,  2.87400000e+03,  2.83700000e+03,  2.83400000e+03,
        5.01000000e+02,  6.42000000e+02,  1.02500000e+03,  1.33700000e+03,
        1.65600000e+03,  1.77500000e+03,  1.79600000e+03,  1.72100000e+03,
        1.84400000e+03,  9.16833333e+02,  5.87444444e+02,  1.04374691e+03,
        1.47811111e+03,  1.75608333e+03,  2.24855556e+03,  2.20650000e+03,
        2.58369444e+03,  2.85097222e+03,  1.20920000e+04,  3.94500000e+03,
        6.01900000e+03,  7.58100000e+03,  1.33400000e+04,  9.71100000e+03,
        1.06430000e+04,  1.56920000e+04,  1.14710000e+04,  6.78141846e+04,
        1.48791117e+04,  2.23658896e+04,  2.79915885e+04,  7.09373637e+04,
        3.52908936e+04,  3.89340798e+04,  8.45719151e+04,  4.01419471e+04,
        3.22149361e+04,  4.13778136e+04,  6.18874046e+04,  7.47903611e+04,
        9.19449728e+04,  1.03579468e+05,  1.06427312e+05,  1.07485890e+05,
        1.15868603e+05,  1.38731479e+03,  1.41614726e+03,  2.44850067e+03,
        3.48381343e+03,  3.98012846e+03,  4.54656600e+03,  5.29431651e+03,
        5.57767559e+03,  6.26726706e+03,  1.11415240e+03,  1.17595650e+03,
        1.99469960e+03,  2.80741320e+03,  3.17789280e+03,  3.62209410e+03,
        4.19320830e+03,  4.47960100e+03,  5.02798340e+03,  1.10936840e+03,
        1.15736500e+03,  1.96988800e+03,  2.78725600e+03,  3.15259800e+03,
        3.58898000e+03,  4.16614700e+03,  4.44711940e+03,  4.99812730e+03,
        2.16316517e+02,  2.79157850e+02,  4.26433609e+02,  5.24352232e+02,
        6.57087245e+02,  7.47788512e+02,  7.79023331e+02,  7.90273584e+02,
        8.62310880e+02,  3.05059441e+04,  2.74788047e+04,  5.09584065e+04,
        7.51943361e+04,  8.41154635e+04,  9.60597043e+04,  1.13913819e+05,
        1.19602105e+05,  1.35274334e+05,  2.14000000e+01,  1.48571429e+01,
        1.85000000e+01,  1.51250000e+01,  2.00000000e+01,  1.38750000e+01,
        4.77777778e+00,  5.59016393e+00,  5.86274510e+00,  5.45714286e+00,
        4.86363636e+00,  4.44791667e+00,  2.34906579e+01,  8.30158730e+00,
        1.76354714e+01,  2.32916667e+01,  2.77503858e+01,  1.21074800e+01,
        3.45485714e+02,  9.33333333e+01,  1.11800000e+02,  9.98333333e+01,
        2.56538462e+02,  1.00509804e+02,  1.93754813e+03,  3.83431742e+02,
        4.67969546e+02,  4.50572639e+02,  1.36418007e+03,  4.57087456e+02,
        3.63401686e+02,  3.54136129e+02,  3.41676240e+02,  3.33925313e+02,
        3.62761844e+02,  3.03244937e+02,  1.06068800e+01,  9.77940674e+00,
        1.03355355e+01,  1.08230580e+01,  1.09969042e+01,  1.05457405e+01,
        8.98057692e+00,  8.49944211e+00,  8.78664500e+00,  9.21130000e+00,
        9.29915000e+00,  8.96010000e+00,  9.07373500e+00,  8.57657895e+00,
        8.81250000e+00,  9.42500000e+00,  9.31612500e+00,  9.14861875e+00,
        4.80118714e+00,  3.16187466e+00,  3.31595406e+00,  3.02019073e+00,
        4.30284116e+00,  2.71242790e+00,  1.84427518e+02,  1.80529612e+02,
        1.78400808e+02,  1.88235066e+02,  1.86849560e+02,  1.90700825e+02,
        6.18773724e+00, -3.83933983e+00,  1.22765025e+00,  1.88706051e+00,
       -2.38529427e+00,  4.52538179e+00, -2.93818386e+00, -2.26383563e+00,
        5.35315437e+00,  6.34748603e+02,  2.30252525e+02,  2.19559852e+02,
       -2.17773019e+02,  2.18273469e+02, -3.37497266e+02, -4.57407541e+02,
       -2.64501478e+02, -3.64820673e+02,  1.00993548e+02,  5.79026956e+01,
        4.25767389e+01, -4.91183568e+01, -5.50145682e+01, -6.59944397e+01,
       -7.03327043e+01, -5.13382529e+01,  6.45852809e+01,  3.70504301e+02,
       -1.27620213e+02,  2.67158500e+02, -1.77331048e+02,  2.31874500e+02,
        1.85216462e+02, -4.28832887e+02,  3.10523931e+02,  4.21855133e+02,
        8.55188571e+03,  2.82060520e+02,  7.04392219e+02, -1.60569143e+03,
        5.22066587e+03, -2.84750204e+03, -5.73568609e+03,  5.57365224e+03,
       -2.44725778e+03,  5.05703839e+04,  1.20462583e+03,  3.23541637e+03,
       -9.77707360e+03,  3.03070344e+04, -1.65879731e+04, -3.27072582e+04,
        3.27092329e+04, -1.38453141e+04,  8.12952364e+03,  6.01029614e+02,
        3.52105323e+03, -4.02499876e+03, -3.47746584e+03, -3.03855017e+03,
        3.83341957e+03, -5.39053725e+03,  3.94023243e+03,  1.73771438e+01,
       -3.60900159e+00,  9.68274324e+00, -5.77969895e+00,  7.41028536e+00,
        7.78012002e+00,  1.33335784e+01,  1.21325112e+01,  1.06389268e+01,
        2.49052849e+01, -4.20201638e+00,  1.41629319e+01, -9.16758280e+00,
        8.41907469e+00, -1.00225152e+01,  1.73109042e+01,  1.82862011e+01,
        1.34983048e+01,  2.67223196e+01, -6.23489242e+00,  1.67494890e+01,
       -9.35610909e+00,  9.99423431e+00, -1.10972597e+01,  2.12938836e+01,
        2.14626757e+01,  1.56230611e+01,  6.85752161e+01, -9.45018524e+00,
        2.11502859e+01, -2.46772840e+01,  3.70223543e+01, -2.09742493e+01,
       -6.06628116e+01,  3.44507662e+01, -2.67594285e+01,  2.35096017e+02,
       -5.24944585e+01,  1.37125758e+02, -1.09483472e+02,  1.07131834e+02,
       -1.35701744e+02,  1.87499333e+02,  1.42184734e+02, -1.11263848e+02,
        1.35066740e-01, -9.97887732e-02,  3.24020969e-02,  7.62954456e-02,
       -1.05593350e-01, -1.88437565e-01,  7.52253634e+00, -5.27087442e+00,
        5.84726423e+00, -6.66546132e+00,  7.75191760e+00,  1.40625000e+01,
        1.59763314e+00,  5.19140047e-01, -5.24556213e-01, -3.84645825e-01,
       -4.54278859e-01,  4.79289941e-01,  9.53479262e+00, -4.33432540e+00,
        4.30764224e+00, -7.38879369e+00,  9.01695373e+00, -1.07540112e+01,
        2.44339592e+02, -9.15187377e+00,  1.15817264e+01, -2.69166667e+01,
        1.00397421e+02, -4.38077237e+01,  1.44486811e+03, -4.08965697e+01,
        5.46633002e+01, -1.61773379e+02,  5.82827586e+02, -2.55199586e+02,
        7.43365646e+01, -1.07536687e+01,  3.30098555e+01, -2.84667126e+01,
       -3.90621111e+01,  6.91980283e+01,  2.59146864e-01, -1.72407464e-01,
        1.01971983e-01,  1.10315250e-01, -2.80478750e-01,  3.24330250e-01,
        3.55753345e-01, -2.12623570e-01,  1.46532878e-01, -1.21960551e-01,
       -3.76743750e-01,  5.00556250e-01,  4.11977324e-01, -2.61979684e-01,
        1.65361467e-01, -1.93515582e-01, -4.16875000e-01,  5.25625000e-01,
        1.95929189e+00, -5.61914477e-01,  4.74719230e-01, -5.23663546e-01,
        7.11968352e-01,  5.74016024e-01,  4.21783394e+00, -1.45938091e+00,
        1.29731981e+00, -9.70448225e-01, -1.41291589e+00, -9.73769883e-01,
       -9.38173548e-01,  7.20851573e-01, -7.21232406e-01, -7.81786466e-01,
       -1.44049125e+00, -7.00677828e-01,  7.77299565e-01, -8.86065685e-01,
       -1.26506024e+00,  2.71084337e+00,  6.80894309e-01, -6.66666667e-01,
       -6.50615901e-01, -5.23502304e-01, -5.57294532e-01, -6.32363584e-01,
        5.66432455e-01, -7.74929668e-01, -1.09929078e+00,  1.24113475e+00,
        4.06961788e-01,  4.48810917e-01, -4.70889176e-01, -1.30534351e+00,
        2.75572519e+00,  3.96733390e-01,  4.50562938e-01, -4.74127321e-01,
       -1.29139933e+00,  2.78876661e+00, -2.72301371e-01,  7.04538523e-01,
       -8.04598091e-01, -7.21070065e-01,  2.40208751e+00, -7.02211378e-01,
        5.89316917e-01, -4.88189854e-01, -1.14238308e+00, -1.35668422e+00,
       -6.96472482e-01,  5.54414835e-01, -5.18329648e-01, -1.26056588e+00,
        1.67483636e+00, -7.21406078e-01,  6.21270569e-01, -5.32878407e-01,
       -1.21937843e+00,  1.53747715e+00, -5.09446480e-01,  7.29855852e-01,
       -1.02023405e+00, -6.92957752e-01, -7.66762827e-01, -9.20250904e-01,
        6.85205673e-01, -5.83965055e-01, -6.23535839e-01,  4.83748635e-01,
        2.04169588e+00,  2.66219446e+00,  1.89710980e+00,  2.07274612e+00,
        2.87936537e+00,  1.41291175e+00,  1.68411528e+00,  1.92022199e+00,
        2.57228916e+00,  2.74342206e+00,  2.01869159e+00,  2.48926117e+00,
        2.19647355e+00,  1.79430673e+00,  1.79603503e+00,  1.73148953e+00,
        1.47903309e+00,  1.58939451e+00,  2.58081879e+00,  3.36101226e+00,
        1.29215465e+00,  1.40574804e+00,  1.76045866e+00,  2.61832061e+00,
        2.42129477e+00,  1.29306943e+00,  1.36238513e+00,  1.73333164e+00,
        2.61168947e+00,  2.41389915e+00,  1.58069803e+00,  1.44285249e+00,
        2.22117534e+00,  1.72980774e+00,  1.84119505e+00,  1.71519188e+00,
        2.25753983e+00,  2.10248031e+00,  2.44141185e+00,  3.33451266e+00,
        1.69495323e+00,  1.69992308e+00,  1.71456926e+00,  2.25082080e+00,
        3.22777722e+00,  1.73974868e+00,  2.00175002e+00,  1.83265821e+00,
        2.16270567e+00,  3.30577656e+00,  2.05955435e+00,  2.03111808e+00,
        2.92999929e+00,  2.03487832e+00,  2.00396449e+00,  2.44351476e+00,
        1.80723204e+00,  2.53788897e+00,  1.68286734e+00,  1.50862405e+00,
        7.77557646e-01, -8.76990091e-01,  7.01786845e+00,  3.86897873e+00,
        4.16603008e+00,  1.83881464e+00,  8.01120462e+00,  1.63914514e+00,
        5.30032732e+01,  5.99557910e+00,  1.26907582e+02,  1.20089132e+01,
        3.25191087e+01,  2.02695205e+01,  4.03467275e+00,  2.63688350e+00,
        4.01113624e+00,  2.44656094e+00,  4.12846365e+00,  2.40725005e+00,
        5.35652353e+00,  1.35999419e+00,  1.74345009e+01,  1.12388070e+01,
        5.22067061e+00,  1.63978497e+03,  8.22505718e+02,  1.20171621e+03,
        1.64488339e+03,  2.17348325e+01,  8.22505718e+02,  7.75000000e+00,
        8.88280466e+00,  4.40106182e-01,  4.28825150e+00,  8.52233457e+02,
        1.03930909e+01,  8.85199465e+00,  1.64006921e+03,  8.22640407e+02,
        1.20190521e+03,  1.64515313e+03,  2.17403216e+01,  8.22640407e+02,
        7.73601197e+00,  8.88281755e+00,  4.40106429e-01,  4.28825295e+00,
        8.52234364e+02,  1.03931020e+01,  8.85199572e+00,  1.96920029e+03,
        9.80091053e+02,  1.45852105e+03,  1.96218755e+03,  2.71877469e+01,
        9.80091053e+02, -1.02961223e+01,  8.89969395e+00,  4.39538741e-01,
        4.29015104e+00,  8.50628343e+02,  1.03735164e+01,  8.85010946e+00,
        1.64135625e+03,  8.23269933e+02,  1.20284745e+03,  1.64641288e+03,
        2.17533447e+01,  8.23269933e+02,  5.50381522e+00,  8.88298004e+00,
        4.40108835e-01,  4.28827125e+00,  8.52192949e+02,  1.03925969e+01,
        8.85194712e+00,  1.62893370e+03,  8.17214874e+02,  1.19385126e+03,
        1.63429495e+03,  2.16147250e+01,  8.17214874e+02,  5.94231946e+00,
        8.88151214e+00,  4.40085183e-01,  4.28810598e+00,  8.52546264e+02,
        1.03969057e+01,  8.85236163e+00,  1.61247496e+03,  8.09290453e+02,
        1.18238909e+03,  1.61843074e+03,  2.13675474e+01,  8.09290453e+02,
        6.68496975e+00,  8.88007822e+00,  4.40052974e-01,  4.28794452e+00,
        8.52781868e+02,  1.03997789e+01,  8.85263795e+00,  2.16464640e+03,
        1.07131006e+03,  1.71253132e+03,  2.14857676e+03,  3.18344211e+01,
        1.07131006e+03, -2.23607345e+01,  8.89473207e+00,  4.38724852e-01,
        4.28959335e+00,  8.54577549e+02,  1.04216774e+01,  8.85474141e+00,
        1.60609854e+03,  8.05256655e+02,  1.17384164e+03,  1.61040306e+03,
        2.18392386e+01,  8.05256655e+02,  5.85473778e+00,  8.87424357e+00,
        4.39991077e-01,  4.28728725e+00,  8.55149949e+02,  1.04286579e+01,
        8.85541099e+00,  2.94016352e+03,  1.79000000e+02,  8.80000000e+01,
        1.68000000e+02,  1.10000000e+01,  2.00000000e+00,  3.50000000e+01,
        3.70000000e+01,  1.68000000e+02,  1.90000000e+01,  5.69367505e-01,
        3.71801046e-01,  2.00000000e+00,  4.00000000e+00,  1.10000000e+01,
        2.10000000e+01,  1.60000000e+01,  1.60000000e+01,  2.40000000e+01,
        8.00000000e+00,  4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        5.77350269e-01,  6.92899286e-01,  9.90385697e-01,  2.16829803e+00,
        4.39952677e+00,  5.77350269e-01,  5.64972123e-01,  8.56308332e-01,
        1.81352679e+00,  3.56159564e+00,  8.10820921e+00,  1.41421356e+00,
        3.07063630e+00,  1.75000000e+00,  8.53347170e+00,  1.75000000e+00,
        2.95172494e+00,  1.58276972e+00,  1.60631694e+01,  3.11620105e+01,
        5.89306340e+01,  9.67051823e+00,  1.82706743e+01,  3.18705741e+01,
        6.15172037e+01,  3.89654521e+01,  3.61487509e+01,  3.03127140e+01,
        2.39563507e+01,  2.15333104e+01,  1.93851206e+01,  1.76451402e+01,
        8.63636364e-01,  5.67513903e-01,  4.50523785e-01,  4.08248290e-01,
        4.85080537e+01,  2.79297150e+01,  2.10684664e+01,  1.64662813e+01,
        1.52310939e+01,  1.33120708e+01,  1.09608711e+01,  8.99579829e+00,
        8.55165968e-01,  6.29603125e-01,  4.22883399e-01,  3.88259715e-01,
        1.04333333e+02,  9.61997336e+01,  9.99693678e+01,  1.80412964e+02,
        1.73160784e+02,  1.76088000e+02,  1.07911206e+02,  2.06723024e+02,
        1.67619048e+00,  1.84800571e+00,  8.32626218e-01,  1.17188638e+00,
        1.15505279e+00,  1.18420000e+00,  1.03013856e+00,  1.20342533e+00,
        2.00990724e+03,  1.00495362e+03,  1.47314553e+03,  2.00990724e+03,
        2.52432505e+01,  1.00495362e+03,  8.87541731e+00,  4.40391741e-01,
        4.28741951e+00,  8.54658415e+02,  1.04226636e+01,  8.85483603e+00,
        1.00000000e+00,  1.40000000e+01,  2.00000000e+00,  1.90000000e+01,
        1.00000000e+00,  1.40000000e+01,  1.70000000e+01,  1.90000000e+01,
        4.00000000e+00,  1.10000000e+01,  1.40000000e+01,  1.40000000e+01,
        5.00000000e+00,  6.00000000e+00,  1.00000000e+00,  2.00000000e+00,
        4.00000000e+00,  2.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        1.10000000e+01,  5.00000000e+00,  4.00000000e+00,  4.00000000e+00,
        3.00000000e+00,  2.00000000e+00,  1.30000000e+01,  8.00000000e+00,
        1.00000000e+01,  2.00000000e+00,  7.00000000e+00,  2.00000000e+00,
        2.00000000e+00,  1.00000000e+00,  3.00000000e+00,  2.00000000e+00,
        1.00000000e+00,  4.00000000e+00,  5.00000000e+00,  2.00000000e+00,
        4.00000000e+00, -1.70880387e+00,  3.03553173e+01,  8.56201970e+00,
        2.24269265e+01,  5.93203388e+00,  3.07423331e+01,  3.44744098e+01,
       -2.76484329e+01,  1.21300496e+01,  1.11441047e+01,  1.24910002e+01,
        4.46008018e+00, -1.00135903e+01,  3.57065768e+01,  2.12601568e+00,
        1.57142999e+01,  1.22973061e+01,  6.32150876e+00,  1.88067345e+01,
        3.86382800e-01,  4.64284369e+01,  2.09037518e+01,  9.60423879e+00,
       -5.22527778e+00,  4.78838151e+00,  2.18899978e+00,  1.46847105e+02,
        8.92170649e+01,  6.39206997e+01,  1.04214522e+01,  9.35792583e+01,
       -1.10698302e+01,  7.76273148e+00,  5.49123959e+00,  3.66046344e+00,
        3.60394788e+00, -1.52896236e+00, -2.18321465e+01,  2.89685086e+01,
        7.43132447e+00,  8.41521293e+00,  4.69700000e+03,  3.74666667e+01,
        6.58531746e-01,  6.59090909e-01,  6.79245283e-01,  4.68750000e-01,
        8.80000000e+01,  1.62500000e+00,  5.25000000e+01,  7.76315789e-01,
        3.55000000e+01,  1.00000000e+00,  4.00000000e+00,  1.66666667e-01,
        1.60472475e+02,  2.19239529e+00,  2.53566102e+01,  4.75683776e-01,
        2.58452693e+02,  3.19077398e+00,  3.89654521e+01,  4.97341565e-01,
        1.23470521e+02,  1.50573807e+00,  1.73590585e+01,  3.26318808e-01,
        2.24412562e+00,  8.31103239e-02,  2.70872935e+00,  8.31103239e-02,
        1.58531746e-01,  1.20000000e-01,  1.06703297e+00,  1.24000000e+00,
        4.56521739e-01,  1.06703297e+00,  1.24000000e+00,  6.41318681e-01,
        1.42452285e-01, -6.41318681e-01,  3.85185185e-01, -3.60000000e+01,
       -6.84523810e-01,  8.70983447e-01,  4.07548387e-01,  1.56983447e-01,
        1.15006000e+03,  6.42857143e-01,  1.90000000e+01,  1.80000000e+01,
        2.39466102e+00,  4.91587378e+00,  5.72640977e+00,  6.42914887e+00,
        6.77264634e+00,  6.90825634e+00,  2.82832878e+02,  5.97760823e+02,
        8.34692430e+02,  9.93786014e+02,  1.04976018e+03,  1.07077973e+03,
        6.10528527e-01,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  5.79380164e-01,  9.29896694e-01,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        5.90374383e+00,  4.41282033e+00,  3.68068532e+00,  3.28489910e+00,
        3.14032078e+00,  2.70658584e+00,  5.99808353e+01,  7.63254422e+01,
        8.30688102e+01,  9.92328102e+01,  1.13736124e+02,  1.13736124e+02,
        2.79468851e+02,  1.56036703e+02,  1.15765165e+02,  8.87728640e+01,
        6.89178149e+01,  6.34477041e+01,  7.90123457e+01,  4.36090488e+01,
        3.11500800e+01,  1.00000000e+00,  1.00000000e+00, -1.45544433e+01,
        9.21370000e+02,  4.75218507e+02,  9.85688743e+01,  3.99901801e+01,
        4.99240473e+01,  3.36711879e+01,  4.04572731e+01,  1.46197857e+02,
        1.26613224e+02,  9.68542204e+01,  1.23208157e+02,  7.37505122e+01,
        4.67516080e+01,  5.89774534e+01,  2.03464762e+01,  9.30381728e+01,
        1.08185672e+01,  3.53045283e+01,  1.01338508e+02,  2.05066098e+02,
        1.10366514e+02,  1.34274099e+02,  4.59960947e+01,  3.78949036e+01,
        2.43154934e+02,  7.25389329e+01,  5.22553233e+01,  1.36848916e+02,
        1.29649118e+02,  2.00905333e+01,  7.54071390e+01,  4.34046491e+01,
        4.59960947e+01,  1.53032217e+02,  1.01823912e+02,  1.06948720e+02,
        9.56254026e+01,  1.03228885e+02,  9.27102463e+01,  1.14338828e+02,
        1.54301653e+02,  6.74784254e+01,  8.07684677e+01,  1.56279060e+02,
        8.84935361e+01,  1.49108905e+02,  3.57065768e+01,  1.17133488e+01,
        3.28761346e+01,  3.83942849e+01,  3.03553173e+01, -2.18321465e+01,
        1.65162182e+02,  2.15328727e+00,  5.37322136e+01,  1.59232955e+00,
        1.20330105e+02,  2.01525298e+00,  3.07874285e+01,  9.95706184e-01,
        3.79423232e+01,  1.10845197e+00,  1.19167381e+01,  5.38042999e-01,
        1.26000000e+02,  1.70000000e+02,  2.74000000e+02,  4.32000000e+02,
        6.44000000e+02,  9.64000000e+02,  1.40400000e+03,  1.94000000e+03,
        2.71800000e+03,  8.74800000e+03,  4.68213123e+00,  5.20948615e+00,
        5.99520753e+00,  6.90775528e+00,  7.79258114e+00,  8.59230646e+00,
        9.38910222e+00,  1.01724057e+01,  1.09189756e+01,  1.15605195e+01,
        1.22437951e+01,  1.80211714e+02,  1.08004286e+02,  1.10000000e+01,
        2.00000000e+00,  1.00000000e+00,  4.00000000e+00,  8.00000000e+00,
        2.00000000e+00,  1.00000000e+00,  1.00000000e+00,  2.00000000e+00,
        7.00000000e+00,  1.00000000e+00,  1.00000000e+00,  4.00000000e+00,
        4.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  8.00000000e+00,  3.00000000e+00,  8.00000000e+00,
        1.00000000e+00,  3.00000000e+00,  3.00000000e+00,  3.00000000e+00,
        9.00000000e+00,  2.00000000e+00,  1.00000000e+00,  4.00000000e+00,
        6.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  7.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        4.00000000e+00,  4.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  2.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  2.00000000e+00,  2.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  2.00000000e+00,  2.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  2.00000000e+00,
        2.00000000e+00,  1.00000000e+00,  1.00000000e+00,  2.00000000e+00,
        2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  2.00000000e+00,  2.80000000e+01,  6.92307692e-01,
        1.26058000e+01,  3.22000400e+02,  5.29050000e+02,  5.29050000e+02,
        2.05000000e+01,  1.20000000e+01,  9.52083333e+00,  8.15277778e+00,
        6.76388889e+00,  4.85818594e+00,  3.54435941e+00,  2.23463089e+00,
        1.88689815e+00,  1.82953372e+00,  6.00000000e-01,  1.85185185e-01,
        1.48809524e-01,  1.03333333e-01,  7.16145833e-02,  6.04081633e-02,
        4.42708333e-02,  3.70370370e-02,  2.67187500e-02,  2.27017651e-02,
        8.47685185e-01,  3.60000000e+01,  1.80000000e+01,  1.00000000e+00,
        5.00000000e-01,  7.45943162e+00,  8.80000000e+01,  6.06145692e+00,
        6.92067150e+00,  7.80751004e+00,  8.78889831e+00,  9.79238838e+00,
        1.07883908e+01,  1.18084939e+01,  1.28288322e+01,  1.38560544e+01,
        2.56230138e+02,  5.17614973e+00,  2.56494936e+00,  6.52356231e+00,
        4.79579055e+00,  8.04462628e+00,  6.94408721e+00,  9.76898426e+00,
        9.06357899e+00,  1.16027205e+01,  1.38810009e+02,  1.15474993e+03,
        2.21910486e+01,  3.92380000e+04,  1.44000000e+02,  4.28000000e+02,
        5.06000000e+02,  3.61388889e+01,  1.89166667e+01])

def mordred_calculate(smiles_right,dirname=''):
    if dirname:
        smi_path=dirname+'/'+'smiles.smi'
        smile_file =open(smi_path,'w')
        for smile in smiles_right:
            smile=smile.replace("'", '')+'\n'
            smile_file.write(smile)
        smile_file.close()
        smi_path=dirname+'/'+'smiles.smi'
        finger_path=dirname+'/'+'finger.csv'
    else:
        smi_path='smiles.smi'
        smile_file =open(smi_path,'w')
        for smile in smiles_right:
            smile=smile.replace("'", '')+'\n'
            smile_file.write(smile)
        smile_file.close()
        smi_path='smiles.smi'
        finger_path='finger.csv'
    mingling = 'python -m mordred ' + smi_path + ' -o ' + finger_path
    os.system(mingling)
    finger=pd.read_csv(finger_path)
    return finger
def load_rfmodel(cutoff):
    if cutoff=='20':
        model_rf0 = pickle.load(open('%s/rf0_h.pkl' % modelpt, 'rb'))
        model_rf1 = pickle.load(open('%s/rf1_h.pkl' % modelpt, 'rb'))
        model_rf2 = pickle.load(open('%s/rf2_h.pkl' % modelpt, 'rb'))
        model_rf3 = pickle.load(open('%s/rf3_h.pkl' % modelpt, 'rb'))
        model_rf4 = pickle.load(open('%s/rf4_h.pkl' % modelpt, 'rb'))
    if cutoff=='50':
        model_rf0 = pickle.load(open('%s/hob_rf0.pkl' % modelpt, 'rb'))
        model_rf1 = pickle.load(open('%s/hob_rf1.pkl' % modelpt, 'rb'))
        model_rf2 = pickle.load(open('%s/hob_rf2.pkl' % modelpt, 'rb'))
        model_rf3 = pickle.load(open('%s/hob_rf3.pkl' % modelpt, 'rb'))
        model_rf4 = pickle.load(open('%s/hob_rf4.pkl' % modelpt, 'rb'))
    return model_rf0,model_rf1,model_rf2,model_rf3,model_rf4

def model_predict(cutoff,smiles_right,mols_right,right_num,modelpt,dirname=''):
    #calc = Calculator(descriptors, ignore_3D=True)
    #mordred_data = calc.pandas(mols_right)
    mordred_data=mordred_calculate(smiles_right)
    pre_list = []
    pre_proba_list1 = []
    pre_proba_list2 = []
    
    pre_data1_0 = mordred_data[save_index1]
    pre_data1 = np.delete(np.array(pre_data1_0), zero1, axis=1)
    pre_data1[np.isnan(pre_data1)] = 0
    model_rf0,model_rf1,model_rf2,model_rf3,model_rf4=load_rfmodel(cutoff)
    rf_pre_proba0 = model_rf0.predict_proba(pre_data1)
    rf_pre_proba1 = model_rf1.predict_proba(pre_data1)
    rf_pre_proba2 = model_rf2.predict_proba(pre_data1)
    rf_pre_proba3 = model_rf3.predict_proba(pre_data1)
    rf_pre_proba4 = model_rf4.predict_proba(pre_data1)

    pre_proba = (rf_pre_proba0 + rf_pre_proba1 + rf_pre_proba2+rf_pre_proba3+rf_pre_proba4)/float(5)

    pre_list=list(np.argmax(pre_proba,axis=1))
    pre_name=['Low','High']
    prediction=[]
    for x in pre_list:
        prediction.append(pre_name[x])
    #print(prediction)
    #pre_list.append(pre)
    pre_proba_list1=[round(a*100,2) for a in pre_proba[:, 0]]
    pre_proba_list2=[round(b*100,2) for b in pre_proba[:, 1]]
    #print(pre_proba_list2)

    pca = joblib.load('pca_hob.m') 
    finger11=pre_data1/avgs
    n_samples = 1157
    newX=pca.transform(finger11)
    print(newX)
    p = Path([(-6.5, 0.5), (0, -4), (5, -2.3), (7.5, -2.5),(11,-1.8),(14,2),(12,4),(8,3.5),(2.5,5.5),(-4,5.5),(-5.5,4),(-6.5,0.5)])
    domain=[]
    for ax in range(newX.shape[0]):
        a = p.contains_point((newX[ax,0], newX[ax,1]))
        domain.append(a)
    
    df_right = pd.DataFrame()
    df_right['num']=right_num
    df_right['smiles'] = smiles_right
    df_right['prediction'] = pre_list
    df_right['HOB Class'] = prediction
    df_right['probability(-)'] = pre_proba_list1
    df_right['probability(+)'] = pre_proba_list2
    df_right['inside the applicability domain'] = domain
    return df_right

modelpt=sys.argv[1]
simles_file=sys.argv[2]
cutoff=sys.argv[3]
with open(simles_file,'r') as smiles_data:
    #smiles_list=smiles_data.readlines()
    content = smiles_data.read()
    #print(content)
    smiles=[smi for smi in re.split('[,;\r\n\s]',content) if smi!='']
    #print(smiles)

mols = [Chem.MolFromSmiles(smi) for smi in smiles]
mols_right = []
smiles_right=[]
right_num = []
mols_false = []
smiles_false=[]
false_num = []
for i in range(len(mols)):
    if mols[i] != None:
        right_num.append(i)
        mols_right.append(mols[i])
        smiles_right.append(smiles[i])
    else:
        false_num.append(i)
        mols_false.append(mols[i])
        smiles_false.append(smiles[i])

df_error=pd.DataFrame()
df_error['num'] = false_num
df_error['smiles']=smiles_false
df_right=model_predict(cutoff,smiles_right, mols_right, right_num,modelpt)
df=pd.concat([df_right,df_error],axis=0)
df=df.sort_values(by='num')
df=df.fillna(-1)
print(df)
df.to_csv('pred_result.csv',index=False)



