

##############################################################
##### Imports
##############################################################


import matplotlib as mpl
import matplotlib.pyplot as plt
#import pymongo
#from pymongo import MongoClient
import numpy as np
from scipy import stats
from scipy.special import binom as binomcoeff
from scipy.optimize import curve_fit
from scipy.integrate import quad
import datetime
import pprint
import math
import os
from matplotlib.ticker import AutoMinorLocator
import subprocess





##############################################################
##### Definitions
##############################################################


# function to return datestring (e.g.: 20190714)
def datestring():
    return str(datetime.datetime.today().year) + str(datetime.datetime.today().month).zfill(2) + str(datetime.datetime.today().day).zfill(2)


# function to return timestring (e.g.: 1725 for 17:25h)
def timestring():
    return str(datetime.datetime.now().time().hour).zfill(2) + str(datetime.datetime.now().time().minute).zfill(2)


# FUNCTION:
#   - read in the data from the text file generated with NEST
#   - delete the file initially generated with NEST
#   - write the NEST data file in a new text file (.txt) (along with the interaction type which is a priori not included in the NEST output file)
#   - return the NEST data in the form of a list tuple which can then be saved as a np.array (.npy)
# INPUT:
#   - interaction_type (string): either "ER" or "NR", will be appended to each line of the output .txt file
#   - begstring (string): name of the initially generated NEST output file
#   - endstring (string): final name of the modified .txt output file
#   - pathstring (string): path in which the initially generated NEST output file is stored
# OUTPUT:
#   - nest_output_tuple_list (list): contains the NEST data in the form of a tuple list (i.e. [(1,2,3), (1,2,3), ...])
def format_NEST_output_txt_file(interaction_type, begstring, endstring, pathstring):
    
    ### extracting NEST data from the initially created NEST file
    # creating the list the NEST output from NEST_output.txt is saved to, containing interaction_type, E_dep, E_drift, N_ph, N_e
    NEST_output_list = []
    # looping over the "NEST_output.txt" file and writing the contents into the upper list
    NEST_output_unmodified_file = open(pathstring +begstring)
    for line in NEST_output_unmodified_file:
        row = line.strip().split("\t")
        if len(row)!=12:
            continue
        elif "E_[keV]" in row[0]:
            continue
        elif "g1" in row[0]:
            continue
        else:
            #print(row)
            NEST_output = [interaction_type, float(row[0]), float(row[1]),  int(row[4]),  int(row[5])]  # order of the NEST output per simulated event: [interaction_type, E [keV],  field [V/cm],  Nph,  Ne]
            NEST_output_list.append(NEST_output)
    NEST_output_unmodified_file.close()
    
    ### deleting the initially created NEST file
    subprocess.call("rm " +pathstring +begstring, shell=True)
    
    ### writing the NEST data to a beautifully formatted .txt. file
    # creating the .txt file the NEST data is saved to
    NEST_output_modified_file = open(pathstring +"/" +endstring +".txt", "w+")
    # writing the relevant data of every single event to file
    NEST_output_modified_file.write("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    NEST_output_modified_file.write("type\tE_dep [keV]\t\tE_drift [V/cm]\tN_ph\t\tN_e\n")
    NEST_output_modified_file.write("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    for i in range(len(NEST_output_list)):
        NEST_output_modified_file.write("{interaction_type}\t\t{energydeposition:.2f}\t\t\t{fieldstrength:.2f}\t\t\t{Nph}\t\t\t{Ne}\r\n".format(interaction_type=str(NEST_output_list[i][0]), energydeposition=(NEST_output_list[i][1]),  fieldstrength=(NEST_output_list[i][2]),  Nph=int(NEST_output_list[i][3]), Ne=int(NEST_output_list[i][4])))
    NEST_output_modified_file.write("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    # closing the file
    NEST_output_modified_file.close()
    
    ### returning the NEST data so it can be saved to a np.array later on
    nest_output_tuple_list = []
    for i in range(len(NEST_output_list)):
        nest_output_tuple_list.append((NEST_output_list[i][0], NEST_output_list[i][1], NEST_output_list[i][2], NEST_output_list[i][3], NEST_output_list[i][4]))
    return nest_output_tuple_list


# FUNCTION:
#   - convert the input tuple list into a np.array (.npy)
#   - save the np.array into the specified folder
# INPUT:
#   - arrayname (string): name of the output np.array (.npy will be appended)
#   - arraypath (string): path into which the array will be saved
#   - arraytuplelist (list): tuple list that will be converted into a np.array
# OUTPUT:
#   none
def nest_output_tuple_to_np_array(arrayname, arraypath, arraytuplelist):
    store_dtype = np.dtype([
        ("interaction_type", np.unicode_, 16),
        ("energy_deposition", np.float64),
        ("field_strength", np.float64),
        ("number_of_photons", np.uint64),
        ("number_of_electrons", np.uint64),
    ])
    nest_output_array = np.array(arraytuplelist, store_dtype)
    np.save(arraypath +arrayname +".npy", nest_output_array)
    return


# function to convert a run_list entry into a string (e.g. for saving the respective NEST run)
def generate_string_from_run_list_entry(run_list_entry):
    run_list_string = "EVENTS_" +str(run_list_entry[0]) +"__INTERACTION_" +str(run_list_entry[1]) +"__ENERGY_" +str(run_list_entry[2]).replace(".","_") +"__EDRIFT_" +str(run_list_entry[3]).replace(".","_") #+".npy"
    return run_list_string


#This function is used to generate a run_list out of the manual input defined within chapter 1.
def get_run_list(flag_manual, flag_sweep, list_manual, p_noe, p_it, p_ed, p_edrift):
    ### run list
    # preparing the run list
    if flag_sweep == True and flag_manual == False:
        run_list = []
        for i in p_noe:
            for j in p_it:
                for k in p_ed:
                    for l in p_edrift:
                        run_list.append([i,j,k,l])
    elif flag_sweep == False and flag_manual == True:
        run_list = list_manual
    elif (flag_sweep == True and flag_manual == True) or (flag_sweep == False and flag_manual == False):
        run_list = []
        print("Both flags (flag_parameter_range and flag_manual_input) are either True or False.")
        print("Make sure you have exactly one flag set to True in order to run the program correctly.")
    else:
        run_list = []
        print("You have picked some strange values for both flags (flag_parameter_range and flag_manual_input).")
        print("They have to be boolean whereas exactly one has to be set to True for the program to work correctly.")
        print("You suck!")
    # printing the run list
    print("run_list = [")
    for i in run_list:
        print("    ", i)
    print("    ]")
    print()
    return run_list


# This function is used to run NEST.
# It takes the run_list (defined within chapter 1) as an input and generates primary quanta out of that.
def run_NEST(runNEST, runlist, runname, nestp, outputp, saveasonearray):
    ### running NEST
    if runNEST == True:
        print("Starting Run: {}\n".format(runname))
        subprocess.call("mkdir " +outputp +runname, shell=True)
        # looping over the run_list and running NEST for every sublist
        arraystringlist = []
        for i in range(len(runlist)):
            savestring = generate_string_from_run_list_entry(run_list_entry=runlist[i])
            #savestring = "EVENTS_" +str(run_list[i][0]) +"__INTERACTION_" +str(run_list[i][1]) +"__ENERGY_" +str(run_list[i][2]).replace(".","_") +"__EDRIFT_" +str(run_list[i][3]).replace(".","_")
            temporarystring = "NEST_output.txt"
            # running nest
            subprocess.call(nestp +" " +str(runlist[i][0]) +" " +str(runlist[i][1]) +" " +str(runlist[i][2]) +" " +str(runlist[i][2]) +" " + str(runlist[i][3]) +" " +"-1" +" >> " +outputp +runname +"/" +temporarystring, shell=True)
            # saving the NEST output as .txt and .npy file
            nest_output_tuple_list = format_NEST_output_txt_file(interaction_type=runlist[i][1], begstring=temporarystring, endstring=savestring, pathstring=outputp+runname+"/")
            nest_output_tuple_to_np_array(arrayname=savestring, arraypath=outputp+runname+"/", arraytuplelist=nest_output_tuple_list)
            arraystringlist.append(savestring +".npy")
            print("The NEST run --->  {}  <--- was saved successfully (.txt and .npy).".format(savestring))
        # summarizing all ndarrays into one single array containing all the data and deleting every single other file
        if saveasonearray==True:
            print(f"\nGenerating Single Output Array: {runname}.npy")
            arraylist = []
            for i in range(len(arraystringlist)):
                arraylist.append(np.load(outputp +runname +"/" +arraystringlist[i]))
            concatenated_array = np.concatenate(arraylist)
            #subprocess.call("rm -r " +outputp +runname +"/*", shell=True)
            for i in range(len(arraystringlist)):
                subprocess.call("rm " +outputp +runname +"/" +arraystringlist[i], shell=True)
                subprocess.call("rm " +outputp +runname +"/" +arraystringlist[i][:-4] +".txt", shell=True)
                #arraylist.append(np.load(outputp +runname +"/" +arraystringlist[i]))
            np.save(outputp +runname +"/" +runname +".npy", concatenated_array)
            
        # end of main program
        print("\nFinished Run: {}".format(runname))
    else:
        print("You chose not to run NEST.")
    return


# This function is used to calculate the root mean square (RMS) of a list of floats
def qmean(data):
    ld = len(data)
    s = np.float128(0)
    for i in range(ld):
        s = np.float128(s +np.multiply(data[i],data[i]), dtype=np.float128)
        #print(f"multiplication: {i}")
        #print(f"{data[i]}*{data[i]}={data[i]*data[i]} ---> s={s}\n")
    return np.float128(np.sqrt(s/ld))


# This function is used to disjunctively seperate an sfndarray into multiple subsets that all share interaction_type, energy_deposition and field_strength
# The returned mask_list contains information on what the values (interaction_type, energy_deposition, drift_field) correspond to the subdatasets.
# The returned subdataset_list contains exactly those subdatasets.
def gen_subdatasets_from_gnampfino_data(data):
    # check for all available subsets
    mask_list = []
    for i in data:
        mask_list_entry = [i["interaction_type"], i["energy_deposition"], i["field_strength"]]
        if mask_list_entry not in mask_list:        
            mask_list.append(mask_list_entry)
    # generate and return a list of subdatasets
    subdataset_list = []
    for i in mask_list:
        subdataset_list.append(data[(data["interaction_type"]==i[0]) & (data["energy_deposition"]==i[1]) & (data["field_strength"]==i[2])])
    return mask_list, subdataset_list


# This function is used to summarize the output .npy files (either multiple ones or one concatenated file if flag_saveasonearray is set to True).
# I.e.: all events with the same properties (i.e. interaction_type, energy_deposition, drift_field) in the end make up one line within the output file that also contains information such as:
# mean, rms, spread, etc...
def gen_summarized_ndarray(outputfolder, runname):
    
    print(f"#############################################################")
    print(f"Starting: Generating Processed ndarray")
    print(f"#############################################################\n")
    
    ### checking the available files
    folder = outputfolder +runname +"/"
    files = [name for name in os.listdir(folder +".") if (os.path.isfile(folder +name) and "__PROCESSED" not in folder+name and ".npy" in folder+name)]
    number_of_files = len(files)

    ### retrieving the subdatasets
    # only one file (that is probably already summarized)
    if number_of_files == 1:
        print(f"Processing Concatenated File: {runname}.npy\n")
        mask_list, subdataset_list = gen_subdatasets_from_gnampfino_data(data=np.load(folder +files[0]))
        print(f"Subsets (interaction_type, energy_deposition, field_strength):")
        for i in range(len(subdataset_list)):
            print(mask_list[i])
    # more than one file (probably because one did not generate a concatenated file)
    elif number_of_files > 1:
        print("There is more than one file.")
        subdataset_list = [np.load(folder +name) for name in os.listdir(folder +".") if (os.path.isfile(folder +name) and "__PROCESSED" not in folder+name and ".npy" in folder+name)]
        print(f"Subsets (interaction_type, energy_deposition, field_strength):")
        for i in range(len(subdataset_list)):
            print("y3a")
            a = subdataset_list[i]['interaction_type'][0]
            b = subdataset_list[i]['energy_deposition'][0]
            c = subdataset_list[i]['field_strength'][0]
            print(f"{a}, {b}, {c}")
    # no file exists
    else:
        print("There is no file in the specified folder.")
        return

    ### processing
    processed_ndarray_tuplelist = []
    for i in range(len(subdataset_list)):
        # defining the dtype
        #print(subdataset_list[i].dtype.names)
        store_dtype = np.dtype([
            ("number_of_events", np.uint64),
            ("interaction_type", np.unicode_, 16),
            ("energy_deposition", np.float64),
            ("field_strength", np.float64),
            ("mean_number_of_photons", np.float64),
            ("rms_number_of_photons", np.float64),
            ("mean_number_of_photons_sigma", np.float64),
            ("mean_number_of_electrons", np.float128),
            ("rms_number_of_electrons", np.float64),
            ("mean_number_of_electrons_sigma", np.float128),
        ])
        # calculations
        number_of_events = len(subdataset_list[i])
        interaction_type = subdataset_list[i]["interaction_type"][0]
        energy_deposition = subdataset_list[i]["energy_deposition"][0]
        field_strength = subdataset_list[i]["field_strength"][0]
        mean_number_of_photons = np.mean(subdataset_list[i]["number_of_photons"])
        rms_number_of_photons = qmean(subdataset_list[i]["number_of_photons"])
        mean_number_of_photons_sigma = (np.std(subdataset_list[i]["number_of_photons"], ddof=1)/np.sqrt(number_of_events))
        mean_number_of_electrons = np.mean(subdataset_list[i]["number_of_electrons"])
        rms_number_of_electrons = qmean(subdataset_list[i]["number_of_electrons"])
        mean_number_of_electrons_sigma = (np.std(subdataset_list[i]["number_of_electrons"], ddof=1)/np.sqrt(number_of_events))
        # appending to the tuple list
        processed_ndarray_tuplelist.append((
            number_of_events,
            interaction_type,
            energy_deposition,
            field_strength,
            mean_number_of_photons,
            rms_number_of_photons,
            mean_number_of_photons_sigma,
            mean_number_of_electrons,
            rms_number_of_electrons,
            mean_number_of_electrons_sigma
        ))

    ### generating the processed ndarray
    processed_ndarray = np.array(processed_ndarray_tuplelist, store_dtype)
    np.save(folder +runname +"__PROCESSED" +".npy", processed_ndarray)
    print(f"Saving Summarized File: {runname}__PROCESSED.npy\n")

    print(f"#############################################################")
    print(f"Finished: Generating Processed ndarray")
    print(f"#############################################################\n")

    return


