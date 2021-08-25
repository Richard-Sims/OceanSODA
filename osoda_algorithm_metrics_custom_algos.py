#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 07:14:07 2021

Calculate algorithm metrics including algorithms for which there are no python implementations (i.e. only model output and std dev data are available)

@author: tom holding
"""

import osoda_global_settings;
import osoda_algorithm_comparison;
import os_algorithms.utilities as utilities;
import os_algorithms.metrics as metrics;
from os_algorithms.diagnostic_plotting import prediction_accuracy_plot;
from os import path, makedirs;
import csv

import numpy as np;
import pandas as pd;
pd.set_option("mode.chained_assignment", None)

import faulthandler;
faulthandler.enable();

runPairedMetrics = False;
runBasicMetrics = True;


#outputVar is AT or DIC etc. definiing the parameter which is the target output of the algorithm
#matchupVariableName is the string name indicating where to find it in the matchup database
#algoRMSD is the goodness of fit from the original algorithm fitting process
#inputUncertainty is the uncertainty associated with the input data
customAlgorithmInfo = [{"name": "ethz_at", #Human readable name, this can be set to anything and is only used as a label
                       "outputVar": "AT", #DIC or AT
                       "matchupVariableName": "ethz_ta_mean", #netCDF variable name of the model output (algorithm prediction)
                       "algoRMSD": 13.0, #netCDF variable name of the RMSD of the (original) algorithm fit
                       "inputUncertaintyName": None, #"ethz_ta_stddev", #propagated input data uncertainty
                       "combinedUncertainty": 21.0, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                       },
                      {"name": "ethz_dic", #Human readable name, this can be set to anything and is only used as a label
                       "outputVar": "DIC", #DIC or AT
                       "matchupVariableName": "ethz_dic_mean", #netCDF variable name of the model output (algorithm prediction)
                       "algoRMSD": 16.3, #netCDF variable name of the RMSD of the (original) algorithm fit
                       "inputUncertaintyName": None, #"ethz_dic_stddev", #propagated input data uncertainty
                       "combinedUncertainty": 16.3, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                       },
                      {"name": "ethz_ph", #Human readable name, this can be set to anything and is only used as a label
                       "outputVar": "region_ph_mean", #DIC or AT
                       "matchupVariableName": "ethz_ph_mean", #netCDF variable name of the model output (algorithm prediction)
                       "algoRMSD": 0.024, #netCDF variable name of the RMSD of the (original) algorithm fit
                       "inputUncertaintyName": None, #"ethz_ph_stddev", #propagated input data uncertainty
                       "combinedUncertainty": 0.024, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                       },
                      {"name": "ethz_pco2", #Human readable name, this can be set to anything and is only used as a label
                       "outputVar": "region_pco2w_mean", #DIC or AT
                       "matchupVariableName": "ethz_pco2_mean", #netCDF variable name of the model output (algorithm prediction)
                       "algoRMSD": 12.0, #netCDF variable name of the RMSD of the (original) algorithm fit
                       "inputUncertaintyName": None, #"ethz_pco2_stddev", #propagated input data uncertainty
                       "combinedUncertainty": 14.0, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                       },
                       {"name": "cmems_ph", #Human readable name, this can be set to anything and is only used as a label
                        "outputVar": "region_ph_mean", #DIC or AT
                        "matchupVariableName": "cmems_ph_mean", #netCDF variable name of the model output (algorithm prediction)
                        "algoRMSD": 0.03, #netCDF variable name of the RMSD of the (original) algorithm fit
                        "inputUncertaintyName": None, #propagated input data uncertainty
                        "combinedUncertainty": 0.03, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                        },
                       {"name": "cmems_pco2", #Human readable name, this can be set to anything and is only used as a label
                        "outputVar": "region_pco2w_mean", #DIC or AT
                        "matchupVariableName": "cmems_pco2_mean", #netCDF variable name of the model output (algorithm prediction)
                        "algoRMSD": 14.8, #netCDF variable name of the RMSD of the (original) algorithm fit
                        "inputUncertaintyName": None, #propagated input data uncertainty
                        "combinedUncertainty": 17.97, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                        },
                       {"name": "pml_at", #Human readable name, this can be set to anything and is only used as a label
                        "outputVar": "AT", #DIC or AT
                        "matchupVariableName": "pml_ta_mu", #netCDF variable name of the model output (algorithm prediction)
                        "algoRMSD": 17, #netCDF variable name of the RMSD of the (original) algorithm fit
                        "inputUncertaintyName": None, #propagated input data uncertainty
                        "combinedUncertainty": 22, #netCDF variable name of the propagated uncertainty combining input data uncertainty with algorithm fit uncertainty
                        },
                    ];

settings = osoda_global_settings.get_default_settings();
outputRoot=path.join("output/example_test_custom_algo_metrics/");
diagnosticPlots = False;
regions = list(settings["regions"]);
sstDatasetName = "SST-ESACCI-OSTIA"; #This is the matchup database variable name corresponding to the SST dataset used as input for the Ethz data
sssDatasetName = "SSS-CORA"; #This is the matchup database variable name corresponding to the SSS dataset used as input for the Ethz data


#First find the combination that corresponds to the Ethz input data.
#We'll then run all algorithms using this input data combination, in order to compare like-for-like.
specificVariableToDatabaseMaps, specificVariableToDatabaseMapNames = utilities.get_dataset_variable_map_combinations(settings);
combinationMap = combinationName = None;
for i, combiName in enumerate(specificVariableToDatabaseMapNames):
    if (sstDatasetName in combiName) and (sssDatasetName in combiName):
        combinationMap = specificVariableToDatabaseMaps[i];
        combinationName = combiName;
        break;
if combinationMap is None:
    raise ValueError("Couldn't find input data combination for the specified SST ({0}) and SSS({1}) data sets.".format(sstDatasetName, sssDatasetName));


if runPairedMetrics == True:
    buildInAlgorithmsMap = settings["algorithmRegionMapping"];
    for region in regions:
        print("Starting regions:", region);
        builtInAlgorithms = settings["algorithmRegionMapping"][region]; #[at_algorithms.Sasse2013_global_at, dic_algorithms.Sasse2013_global_dic]; 
        customAlgorithms = [cai for cai in customAlgorithmInfo if cai["outputVar"] in ["AT", "DIC"]];
        
        
        #### Create the directory to contain outputs. This is structured to be analogous to the main() output, even though it only contains a single input combination run
        currentCombinationOutputDirectory = path.join(outputRoot, combinationName);
        if path.exists(currentCombinationOutputDirectory) == False:
            makedirs(currentCombinationOutputDirectory);
        
        #### Write information about the combination of input datasets used:
        utilities.write_specific_variable_to_database_mapping(combinationMap, path.join(currentCombinationOutputDirectory, "inputs_used.txt"), combinationName);
        
        
        ### #### The non-custom algorithms need to be ran on the matchup database's input data for the selected input combination
        #### So this data must be extracted
        years = utilities.calculate_years_for_input_combination(settings, combinationMap); #Find the years where there is overlap in the selected input data combinations
        matchupData = utilities.load_matchup_to_dataframe(settings, combinationMap, years=years); #each year is concatinated to create a single dataframe
        
         
        SST_max=40;
        SST_min=-10;
        SSS_max=50;
        SSS_min=0;      
        DIC_max=3000;
        DIC_min=500; 
        pH_max=8.5;
        pH_min=7;
        pCO2_max=800;
        pCO2_min=100;
        TA_max=3000;
        TA_min=500;
        
        #SST
        mdb_SST = matchupData["SST"];
        mdb_SST_numeric=mdb_SST.values-273.15;
        #Upper realistic limit 
        states=mdb_SST_numeric>SST_max;
        index_temp_exceed=np.where(states)[0]
        #Lower realistic limit -10 degrees
        states2=mdb_SST_numeric<SST_min;
        index_temp_below=np.where(states2)[0]
        
        #Salinity
        mdb_SSS = matchupData["SSS"];
        mdb_SSS_numeric=mdb_SSS.values;
        #Upper realistic limit 50 PSU
        states3=mdb_SSS_numeric>SSS_max;
        index_sal_exceed=np.where(states3)[0]
        #Lower realistic limit <0 PSU
        states4=mdb_SSS_numeric<SSS_min;
        index_sal_below=np.where(states4)[0]        
        
        #DIC
        mdb_DIC = matchupData["DIC"];
        mdb_DIC_numeric=mdb_DIC.values;
        #Upper realistic limit 2500 UMOLKG?
        states5=mdb_DIC_numeric>DIC_max;
        index_DIC_exceed=np.where(states5)[0]
        #Lower realistic limit <500 UMOL KG
        states6=mdb_DIC_numeric<DIC_min;
        index_DIC_below=np.where(states6)[0] 
        
        #pH
        mdb_pH = matchupData["region_ph_mean"];
        mdb_pH_numeric=mdb_pH.values;
        #Upper realistic limit 8.5
        states7=mdb_pH_numeric>pH_max;
        index_pH_exceed=np.where(states7)[0]
        #Lower realistic limit <7
        states8=mdb_pH_numeric<pH_min;
        index_pH_below=np.where(states8)[0]
        
        #pCO2
        mdb_pco2 = matchupData["region_pco2w_mean"];
        mdb_pco2_numeric=mdb_pco2.values;
        #Upper realistic limit 700 ppm
        states9=mdb_pco2_numeric>pCO2_max;
        index_pco2_exceed=np.where(states9)[0]
        #Lower realistic limit <200 ppm
        states10=mdb_pco2_numeric<pCO2_min;
        index_pco2_below=np.where(states10)[0]
        
        #TA
        mdb_TA = matchupData["AT"];
        mdb_TA_numeric=mdb_TA.values;
        #Upper realistic limit 3000 umol kg
        states11=mdb_TA_numeric>TA_max;
        index_TA_exceed=np.where(states11)[0]
        #Lower realistic limit <500 umol kg
        states12=mdb_TA_numeric<TA_min;
        index_TA_below=np.where(states12)[0]        
        

        #these variables could also have bounds placed on them but not applied 
        #for this iteration
        # lat long date OC chla DO NO3 PO4 SiO4
        
        # now produce a file with all of the out of bounds data points from the mdb
        
        mdb_flag_warnings_list=['SST greater than maximum value of', 'SST less than minimum value of', 'SSS greater than maximum value of', 'SST less than minimum value of'\
                                , 'DIC greater than maximum value of', 'DIC less than minimum value of','pH greater than maximum value of','pH less than minimum value of'\
                                ,'pCO2 greater than maximum value of','pCO2 less than minimum value of','TA greater than maximum value of','TA less than minimum value of']

        mdb_flag_index_list=[index_temp_exceed, index_temp_below, index_sal_exceed, index_sal_below, index_DIC_exceed, index_DIC_below\
                             ,index_pH_exceed,index_pH_below,index_pco2_exceed,index_pco2_below,index_TA_exceed,index_TA_below]
        
            
        mdb_flag_limits_list=[SST_max,SST_min,SSS_max,SSS_min,DIC_max, DIC_min,  pH_max, pH_min,pCO2_max, pCO2_min, TA_max,TA_min]  
        
        for idx, g in enumerate(mdb_flag_index_list):
            print(idx, g)
            if len(g) == 0:
                print("list is empty")
            else:
                #this prints a header for what the entries have been flagged for
                with open('mdb_flag.csv','a') as result_file:
                    wr = csv.writer(result_file, dialect='excel')
                    wr.writerow([mdb_flag_warnings_list[idx]]+ [mdb_flag_limits_list[idx]])
                #this prints the values that have been flagged to the csv    
                print(matchupData.loc[g].to_csv("mdb_flag.csv",mode='a'));
       
        #now filter those mdb from the analysis
        mdb_ind_rmv =np.concatenate(mdb_flag_index_list) #combine all the numpy arrays into a single array of 'bad data point indexes'
        matchupData = matchupData.drop(matchupData.index[mdb_ind_rmv])

        ### Run each of the build in algorithms to compute model output, extract RMSD etc
        allAlgoOutputInfo = [];
        for ialgorithm, AlgorithmFunctor in enumerate(builtInAlgorithms):
            print("Calculating model outputs for implemented algorithms ({0}/{1}): {2}".format(ialgorithm+1, len(builtInAlgorithms), str(AlgorithmFunctor)));
            algorithm = AlgorithmFunctor(settings);
            try:
                modelOutput, propagatedInputUncertainty, rmsd, combinedUncertainty, dataUsedIndices, dataUsed = \
                          osoda_algorithm_comparison.run_algorithm(algorithm, matchupData,
                                                                   regionMaskPath=settings["regionMasksPath"], region=region,
                                                                   useDepthMask=settings["subsetWithDepthMask"],
                                                                   depthMaskPath=settings["depthMaskPath"],
                                                                   depthMaskVar=settings["depthMaskVar"],
                                                                   useDistToCoastMask=settings["subsetWithDistToCoast"],
                                                                   distToCoastMaskPath=settings["distToCoastMaskPath"],
                                                                   distToCoastMaskVar=settings["distToCoastMaskVar"]
                                                                   );
                
                ### Store the algorithm output information in a dictionary, and then append it to the list
                algorithmOutput = {};
                algorithmOutput["instance"] = algorithm;
                algorithmOutput["name"] = algorithm.__str__();
                algorithmOutput["outputVar"] = algorithm.output_name();
                algorithmOutput["modelOutput"] = modelOutput;
                algorithmOutput["propagatedInputUncertainty"] = propagatedInputUncertainty;
                algorithmOutput["rmsd"] = rmsd;
                algorithmOutput["combinedUncertainty"] = combinedUncertainty;
                algorithmOutput["dataUsedIndices"] = dataUsedIndices;
                allAlgoOutputInfo.append(algorithmOutput);
                
                if diagnosticPlots == True:
                    outputVariable = algorithm.output_name();
                    savePath = path.join(outputRoot, "diagnostic_plots", algorithm.__str__().split(":")[0]+".png");
                    prediction_accuracy_plot(dataUsed[outputVariable], modelOutput, algorithm.__class__.__name__, outputVariable, savePath=savePath);
                
                print("Output calculated (region:"+region+", algo: "+algorithm.__class__.__name__+")");
            except ValueError as e: #Raised if there are no matchup data rows left after spatial mask and algorithm internal subsetting has taken place
                print(algorithm, e.args[0]);
                print("No matchup data left after subsettings (region:"+region+", algo: "+algorithm.__class__.__name__+")");
                continue;
                
                
        ###### Now add extract the custom algorithm data (e.g. rmsd, output), construct output dictionaries, and append them to the algo output info list.
        for customAlgo in customAlgorithms:
            print("Extracting data for custom algorithm: {0}".format(customAlgo["name"]));
            
            #Extract the model output from the matchup data
            colsToExtract = ["date", customAlgo["outputVar"], customAlgo["matchupVariableName"]];
            ##### TODO:
            ##### Missing: combined uncertainty, matchupRMSD, matchupBias?
            customAlgoData = utilities.read_matchup_cols(settings["matchupDatasetTemplate"], colsToExtract, years); #returns data frame containing data from the matchup database for each variable in 'cols'
            ###Subset to remove where model data is NaN
            customAlgoData = customAlgoData.loc[np.isfinite(customAlgoData[customAlgo["matchupVariableName"]])]; #remove where there is no model predictions
            customAlgoData = customAlgoData.loc[np.isfinite(customAlgoData[customAlgo["outputVar"]])]; #remove where there is no reference outputVar data
            
            
            algorithmOutput = {};
            algorithmOutput["instance"] = None;
            algorithmOutput["name"] = customAlgo["name"];
            algorithmOutput["outputVar"] = customAlgo["outputVar"]; #e.g. AT or DIC
            algorithmOutput["modelOutput"] = customAlgoData[customAlgo["matchupVariableName"]]; #The modelled output DIC or AT from matchup database
            algorithmOutput["propagatedInputUncertainty"] = None;
            algorithmOutput["rmsd"] = customAlgo["algoRMSD"]; #Goodness of fit from the original algorithm fit
            algorithmOutput["combinedUncertainty"] = customAlgo["combinedUncertainty"];
            algorithmOutput["dataUsedIndices"] = customAlgoData.index;
            allAlgoOutputInfo.append(algorithmOutput);
            
            if diagnosticPlots == True:
                savePath = path.join(outputRoot, "diagnostic_plots", customAlgo["name"]+".png");
                prediction_accuracy_plot(customAlgoData[customAlgo["outputVar"]], customAlgoData[customAlgo["matchupVariableName"]], customAlgo["name"], customAlgo["outputVar"], savePath=savePath);
            
        ####Calculate metrics
        for currentOutputVar in ["AT", "DIC"]:
            ##Subset the algorithm output data by the target output variable
            algorithmOutputGroup = [algorithmOutput for algorithmOutput in allAlgoOutputInfo if algorithmOutput["outputVar"]==currentOutputVar];
            
            
            basicMetrics, nIntersectMatrix, pairedScoreMatrix, pairedWScoreMatrix, pairedRmsdMatrix, pairedWRmsdMatrix, finalScores = \
                            metrics.calc_all_metrics(algorithmOutputGroup, matchupData, settings);
            print("Completed metrics for", currentOutputVar);
            
            ###########################
            ### Write outputs to file #
            outputDirectory = path.join(currentCombinationOutputDirectory, currentOutputVar, region);
            print("Writing metrics to: ", outputDirectory);
            osoda_algorithm_comparison.write_metrics_to_file(outputDirectory, matchupData, basicMetrics, nIntersectMatrix, pairedScoreMatrix, pairedWScoreMatrix, pairedRmsdMatrix, pairedWRmsdMatrix, finalScores, algorithmOutputGroup);
        
        
        #Calculate summary table for weighted metrics and output to file
        summaryTable_weighted = osoda_algorithm_comparison.create_summary_table(settings, [combinationName], [combinationMap], useWeighted=True, regionOverload = regions);
        summaryTableOutputPath = path.join(outputRoot, combinationName, "summary_best_algos.csv");
        summaryTable_weighted.to_csv(summaryTableOutputPath, sep=",", index=False);     
        print("Full weighted summary table written to:", path.abspath(summaryTableOutputPath));
           
        #Calculate summary table for unweighted metrics and output to fill
        summaryTable_unweighted = osoda_algorithm_comparison.create_summary_table(settings, [combinationName], [combinationMap], useWeighted=False, regionOverload = regions);
        summaryTableOutputPathUnweighted = path.join(outputRoot, combinationName, "summary_best_algos_unweighted.csv");
        summaryTable_unweighted.to_csv(summaryTableOutputPathUnweighted, sep=",", index=False);     
        print("Full unweighted summary table written to:", path.abspath(summaryTableOutputPathUnweighted));





if runBasicMetrics == True:
    ##########
    #calculate individual level metrics for the non-AT/DIC parameters
    import os_algorithms.metrics as metrics;
    import pickle;
    import json;
    
    
    years = utilities.calculate_years_for_input_combination(settings, combinationMap);
    matchupData = utilities.load_matchup_to_dataframe(settings, combinationMap, years); #each year is concatinated to create a single dataframe
    
    
    SST_max=40;
    SST_min=-10;
    SSS_max=50;
    SSS_min=0;      
    DIC_max=3000;
    DIC_min=500; 
    pH_max=8.5;
    pH_min=7;
    pCO2_max=800;
    pCO2_min=100;
    TA_max=3000;
    TA_min=500;
    
    #SST
    mdb_SST = matchupData["SST"];
    mdb_SST_numeric=mdb_SST.values-273.15;
    #Upper realistic limit 
    states=mdb_SST_numeric>SST_max;
    index_temp_exceed=np.where(states)[0]
    #Lower realistic limit -10 degrees
    states2=mdb_SST_numeric<SST_min;
    index_temp_below=np.where(states2)[0]
    
    #Salinity
    mdb_SSS = matchupData["SSS"];
    mdb_SSS_numeric=mdb_SSS.values;
    #Upper realistic limit 50 PSU
    states3=mdb_SSS_numeric>SSS_max;
    index_sal_exceed=np.where(states3)[0]
    #Lower realistic limit <0 PSU
    states4=mdb_SSS_numeric<SSS_min;
    index_sal_below=np.where(states4)[0]        
    
    #DIC
    mdb_DIC = matchupData["DIC"];
    mdb_DIC_numeric=mdb_DIC.values;
    #Upper realistic limit 2500 UMOLKG?
    states5=mdb_DIC_numeric>DIC_max;
    index_DIC_exceed=np.where(states5)[0]
    #Lower realistic limit <500 UMOL KG
    states6=mdb_DIC_numeric<DIC_min;
    index_DIC_below=np.where(states6)[0] 
    
    #pH
    mdb_pH = matchupData["region_ph_mean"];
    mdb_pH_numeric=mdb_pH.values;
    #Upper realistic limit 8.5
    states7=mdb_pH_numeric>pH_max;
    index_pH_exceed=np.where(states7)[0]
    #Lower realistic limit <7
    states8=mdb_pH_numeric<pH_min;
    index_pH_below=np.where(states8)[0]
    
    #pCO2
    mdb_pco2 = matchupData["region_pco2w_mean"];
    mdb_pco2_numeric=mdb_pco2.values;
    #Upper realistic limit 700 ppm
    states9=mdb_pco2_numeric>pCO2_max;
    index_pco2_exceed=np.where(states9)[0]
    #Lower realistic limit <200 ppm
    states10=mdb_pco2_numeric<pCO2_min;
    index_pco2_below=np.where(states10)[0]
    
    #TA
    mdb_TA = matchupData["AT"];
    mdb_TA_numeric=mdb_TA.values;
    #Upper realistic limit 3000 umol kg
    states11=mdb_TA_numeric>TA_max;
    index_TA_exceed=np.where(states11)[0]
    #Lower realistic limit <500 umol kg
    states12=mdb_TA_numeric<TA_min;
    index_TA_below=np.where(states12)[0]        
    

    #these variables could also have bounds placed on them but not applied 
    #for this iteration
    # lat long date OC chla DO NO3 PO4 SiO4
    
    # now produce a file with all of the out of bounds data points from the mdb
    
    mdb_flag_warnings_list=['SST greater than maximum value of', 'SST less than minimum value of', 'SSS greater than maximum value of', 'SST less than minimum value of'\
                            , 'DIC greater than maximum value of', 'DIC less than minimum value of','pH greater than maximum value of','pH less than minimum value of'\
                            ,'pCO2 greater than maximum value of','pCO2 less than minimum value of','TA greater than maximum value of','TA less than minimum value of']

    mdb_flag_index_list=[index_temp_exceed, index_temp_below, index_sal_exceed, index_sal_below, index_DIC_exceed, index_DIC_below\
                         ,index_pH_exceed,index_pH_below,index_pco2_exceed,index_pco2_below,index_TA_exceed,index_TA_below]
    
        
    mdb_flag_limits_list=[SST_max,SST_min,SSS_max,SSS_min,DIC_max, DIC_min,  pH_max, pH_min,pCO2_max, pCO2_min, TA_max,TA_min]  
    
    for idx, g in enumerate(mdb_flag_index_list):
        print(idx, g)
        if len(g) == 0:
            print("list is empty")
        else:
            #this prints a header for what the entries have been flagged for
            with open('mdb_flag.csv','a') as result_file:
                wr = csv.writer(result_file, dialect='excel')
                wr.writerow([mdb_flag_warnings_list[idx]]+ [mdb_flag_limits_list[idx]])
            #this prints the values that have been flagged to the csv    
            print(matchupData.loc[g].to_csv("mdb_flag.csv",mode='a'));

    #now filter those mdb from the analysis
    mdb_ind_rmv =np.concatenate(mdb_flag_index_list) #combine all the numpy arrays into a single array of 'bad data point indexes'
    
    #matchupData = matchupData.drop(matchupData.index[mdb_ind_rmv])

    #Extract algorithm output for each custom algorithm
    algorithmOutputs = [];
    dataUsedList = [];
    for customAlgo in customAlgorithmInfo:
        print("Extracting data for custom algorithm: {0}".format(customAlgo["name"]));
            
        #Extract the model output from the matchup data
        colsToExtract = ["date", "lat", customAlgo["outputVar"], customAlgo["matchupVariableName"]];
        
        ##### TODO:
        ##### Missing: combined uncertainty, matchupRMSD, matchupBias?
        customAlgoData = utilities.read_matchup_cols(settings["matchupDatasetTemplate"], colsToExtract, years); #returns data frame containing data from the matchup database for each variable in 'cols'
        
        # data are loaded in here again, delete rows that are removed by QC
        A=customAlgoData.index#these are the rows in the subset
        B=mdb_ind_rmv#these are the bad rows to removed
        #this finds the indexes
        B_unique_sorted, B_idx = np.unique(B, return_index=True)
        B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
        inthesubset=B_idx[B_in_A_bool]
        C=B[inthesubset]
        #this line then drops the bad rows
        customAlgoData = customAlgoData.drop(customAlgoData.index[inthesubset]); #remove where there is no reference outputVar data


        ###Subset to remove where model data is NaN
        customAlgoData = customAlgoData.loc[np.isfinite(customAlgoData[customAlgo["matchupVariableName"]])]; #remove where there is no model predictions
        customAlgoData = customAlgoData.loc[np.isfinite(customAlgoData[customAlgo["outputVar"]])]; #remove where there is no reference outputVar data
        if customAlgo["name"] == "cmems_pco2": #unit conversion
            customAlgoData["cmems_pco2_mean"] = customAlgoData["cmems_pco2_mean"]*0.00000986923 * 1000000;
            
        
        algorithmOutput = {};
        algorithmOutput["instance"] = None;
        algorithmOutput["name"] = customAlgo["name"];
        algorithmOutput["outputVar"] = customAlgo["outputVar"]; #e.g. AT or DIC
        algorithmOutput["modelOutput"] = customAlgoData[customAlgo["matchupVariableName"]]; #The modelled output DIC or AT from matchup database
        algorithmOutput["propagatedInputUncertainty"] = None;
        algorithmOutput["rmsd"] = customAlgo["algoRMSD"]; #Goodness of fit from the original algorithm fit
        algorithmOutput["combinedUncertainty"] = customAlgo["combinedUncertainty"];
        algorithmOutput["dataUsedIndices"] = customAlgoData.index;
        algorithmOutputs.append(algorithmOutput);
        
        dataUsedList.append(matchupData.iloc[algorithmOutput["dataUsedIndices"]]);
        
        if diagnosticPlots == True:
            savePath = path.join(outputRoot, "diagnostic_plots", customAlgo["name"]+".png");
            prediction_accuracy_plot(customAlgoData[customAlgo["outputVar"]], customAlgoData[customAlgo["matchupVariableName"]], customAlgo["name"], customAlgo["outputVar"], savePath=savePath);
    
    
    basicMetricsMap = {};
    for i in range(0, len(algorithmOutputs)):
        basicMetrics = metrics.calc_basic_metrics(algorithmOutputs[i], dataUsedList[i], settings);
        basicMetricsMap[algorithmOutputs[i]["name"]] = basicMetrics;
    
    #####Write basicMetrics object
    if path.exists(outputRoot) == False:
        makedirs(outputRoot);
        
    #binary pickle dump
    pickle.dump(basicMetricsMap, open(path.join(outputRoot, "basic_metrics_ETHZ.pickle"), 'wb'));
    #basicMetricsRead = pickle.load(open(path.join(outputRoot, "basic_metrics_ETHZ.pickle"), 'rb'));
    
    #For json file, remove long vectors
    ##For json files, convert pd.Series types to nunpy array for json serialisation
    for key in basicMetricsMap.keys():
        del basicMetricsMap[key]["model_output"];
        del basicMetricsMap[key]["reference_output_uncertainty"];
        del basicMetricsMap[key]["weights"];
        del basicMetricsMap[key]["model_uncertainty"];
 
    
    with open(path.join(outputRoot, "basic_metrics_ETHZ.json"), 'w') as file:
        json.dump(basicMetricsMap, file, indent=4);

    
