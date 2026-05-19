library(data.table)
library(doParallel)
library(TSA)
library(signal)
library(readxl)
library(stringr)
source("functions public.R")


#DATA_DIR = "d:/data/2019.05 - sem upd"
DATA_DIR = "."

ic_only_in_eog_artifacts = 1

in_dir = paste0(DATA_DIR, "/preprocessed segments")
out_dir = paste0(DATA_DIR, "/post-ica segments")
file_artifacts = "artifacts.txt"



files = list.files(in_dir, pattern="*.txt", full.names=T)


#Don't change the filter, all thresholds depend on it
bf15 = butter(2, 15/125, "low") #the real cutoff frequency will be higher, approx 3/2 times the stated freq
bf7  = butter(2, 7/125, "low")  #the real cutoff frequency will be higher, approx 3/2 times the stated freq


icas = data.table(read_excel("icas.xlsx"))
thresholds = data.table(read_excel("artifact eog.xlsx"), key="Subject")


artf_agr = foreach (file=files, .packages=c("data.table", "signal", "stringr"), .noexport=c(), .verbose=F) %do% {
  dat = fread(file, stringsAsFactors=T, encoding="UTF-8",
              select=c(1:8), sep="\t",
              col.names=c("Subject", "exp_ver", "Item", "word_pos", "dp", "electrode", "uV", "noun_pos"),
              colClasses=list(character=c(1,3,4,6), integer=c(2,5,7,8)))
  id_subject = dat[1, Subject]
  
  dat[, dp:=dp-2] #shift by 8ms due to the delay of displaying stimuli on CRT monitors (60Hz). 
  
  segments = data.table(Item=unique(dat$Item))
  segments[, n_segment:=1:nrow(segments)]
  dat = merge(dat, segments, by=c("Item"))
  rm (segments)
  setkeyv(dat, c("Subject", "Item", "word_pos", "electrode", "dp"))
  
  id_cols = c("Subject", "exp_ver", "Item", "word_pos", "dp", "n_segment", "noun_pos")
  
  dat[word_pos=="adj", word_pos:="1"]
  dat[word_pos=="noun", word_pos:="0"]
  dat[, word_pos:=as.integer(as.character(word_pos))]
  
  dat2 = dat[dp < 0, .(baseline=mean(uV)), by=.(Subject, Item, word_pos, electrode)]
  dat = merge(dat, dat2, by=c("Subject", "Item", "word_pos", "electrode"))
  dat[, uV_bline:=uV-baseline]
  dat[, baseline:=NULL]
  rm (dat2)
  
  
  if (ic_only_in_eog_artifacts == 1) {
    #low-pass filter
    dat[, uV_filt:=filtruj(bf15, uV_bline), by=.(Subject, Item, word_pos, electrode)] #prefilter for eog thresholds
    
    
    dt_wide = dcast(dat[, c(id_cols, "electrode", "uV_filt"), with=F], ... ~ electrode, value.var="uV_filt")
    
    
    #Step detect in IC space
    dt_wide_ic = uV_to_IC(dt_wide)
    heog_ic = icas[Subject==id_subject, HEOG]
    heog_ic = as.numeric(str_split(heog_ic, " ")[[1]][1])
    veog_ic = icas[Subject==id_subject, blinks]
    veog_ic = as.numeric(str_split(veog_ic, " ")[[1]][1])
    ret = dt_wide_ic[, .(max_step_heog=step_detect(.SD, electrode=sprintf("V%d", heog_ic+1), window_len_dp=10, step_size_dp=1),
                         max_step_veog=step_detect(.SD, electrode=sprintf("V%d", veog_ic+1), window_len_dp=20, step_size_dp=1)), 
                     by=.(Subject, Item, word_pos, n_segment)]  #40ms step, 80 ms one-step window
    
    
    # Check how many segments would be removed due to EOG criteria
    ret2 = ret[as.numeric(as.character(Item)) < 400]
    heog_thr = thresholds[.(id_subject), heog_threshold] #find by key
    veog_thr = thresholds[.(id_subject), veog_threshold] #find by key
    heog_mustremove_thr = thresholds[.(id_subject), heog_threshold_mustremove]
    items_mustremove = str_split(thresholds[.(id_subject), items_mustremove], ";")[[1]]
    
    percent_removed = ret2[max_step_heog > heog_thr | 
                             max_step_veog > veog_thr | 
                             Item %in% items_mustremove | 
                             max_step_heog > heog_mustremove_thr, 
                           .N] / nrow(ret2)
    percent_removed_heog = ret2[max_step_heog > heog_thr | max_step_heog > heog_mustremove_thr, .N] / nrow(ret2)
    percent_removed_veog = ret2[max_step_veog > veog_thr, .N] / nrow(ret2)
    percent_items = ret2[Item %in% items_mustremove, .N] / nrow(ret2)
    cat(sprintf("%s\t%.02f, just heog: %.02f, just veog: %.02f, selected items: %.02f\n", id_subject, percent_removed, percent_removed_heog, percent_removed_veog, percent_items))
    
    
    #Mark segments to remove or to correct
    ret[, action:="stay"]
    
    #For participants with >30% EOG artifacts, we correct them instead of removing entire segments
    if (percent_removed > 0.3) {
      ret[max_step_heog > heog_thr | max_step_veog > veog_thr, action:="correct"]
    } else {
      ret[max_step_heog > heog_thr | max_step_veog > veog_thr, action:="del"]
    }
    ret[Item %in% items_mustremove | max_step_heog > heog_mustremove_thr, action:="del"] #this line must be after other criteria to override them
  } else {
    ret = unique(dat[, .(Subject, Item, word_pos)])
    ret[, action:="correct"]
  }
  
  
  # re-ICA data, this time with no filtering or baseline
  dt_wide = dcast(dat[, c(id_cols, "electrode", "uV"), with=F], ... ~ electrode, value.var="uV")
  dt_wide_ic = uV_to_IC(dt_wide)
  
  # Attach action data, delete some items
  dt_wide_ic = merge(dt_wide_ic, ret[, .(Item, Subject, word_pos, action)], by=c("Subject", "Item", "word_pos"))
  dt_wide_ic = dt_wide_ic[action!="del"]
  
  #Identify ICs to filter or remove
  ic_to_remove = icas[Subject==id_subject, `single el. artifacts`]
  ic_to_remove = str_split(paste(ic_to_remove, collapse=" "), " ")[[1]]
  ic_to_remove = as.numeric(ic_to_remove[ic_to_remove!="NA"]) + 1#component 00 is first column in inverse matrix
  ic_to_remove = ic_to_remove[!is.na(ic_to_remove)]
  ic_to_filter = icas[Subject==id_subject, muscle]
  ic_to_filter = str_split(ic_to_filter, " ")[[1]]
  ic_to_filter = as.numeric(ic_to_filter[ic_to_filter!="NA"]) + 1#component 00 is first column in inverse matrix
  ic_to_filter = ic_to_filter[!is.na(ic_to_filter)]
  ic_to_filter = ic_to_filter[!ic_to_filter %in% ic_to_remove] #if zeroed, don't filter
  
  #Filter selected ICA channels
  for (ic in ic_to_filter) {
    ic_col = sprintf("V%d", ic)
    setnames(dt_wide_ic, ic_col, "ic_to_filter")
    dt_wide_ic[, ic_to_filter:=filtruj(bf15, ic_to_filter), by=.(Subject, Item)]
    setnames(dt_wide_ic, "ic_to_filter", ic_col)
  }
  
  #In all trials remove single-electrode ICs
  a_fwd = dt_wide_ic[, which(colnames(dt_wide_ic)=="V1"):ncol(dt_wide_ic)]
  if (length(ic_to_remove)) {
    a_fwd[, (ic_to_remove):=0]
  }
  
  #Remove EOG channels if correction needed
  ic_to_remove = icas[Subject==id_subject, c(blinks, HEOG)]
  ic_to_remove = str_split(paste(ic_to_remove, collapse=" "), " ")[[1]]
  ic_to_remove = as.numeric(ic_to_remove[ic_to_remove!="NA"]) + 1#component 00 is first column in inverse matrix
  ic_to_remove = ic_to_remove[!is.na(ic_to_remove)]
  if (length(ic_to_remove)) {
    a_fwd[action=="correct", (ic_to_remove):=0] #correct only trials marked as ones that need correction
  }
  
  #Convert back to electrode space
  inv = fread(sprintf("ica/%s.ica.inv.txt", id_subject))
  inv_electrodes = inv[,V1]
  inv[, V1:=NULL]
  inv = as.matrix(inv)
  a_inv = as.matrix(a_fwd[,1:(ncol(a_fwd)-1)]) %*% t(inv)
  colnames(a_inv) = inv_electrodes
  
  dt_wide = dt_wide_ic[, id_cols, with=F]
  dt_wide = cbind(dt_wide, a_inv)
  
  dat = melt(dt_wide, id.vars=id_cols)
  setnames(dat, c("variable", "value"), c("electrode", "uV"))
  
  
  # Re-refer to A1+A2
  dat_a2 = dat[electrode=="A2", .(Subject, Item, word_pos, dp, A2=uV)]
  dat = merge(dat, dat_a2, by=c("Subject", "Item", "word_pos", "dp"))
  dat[electrode != "A2", uV:=uV-(A2/2)]
  dat[, A2:=NULL] #re-refence in artifacts too????
  
  dat[, uV:=round(uV)]
  
  #All further computations will be based on this data
  fwrite(dat, sprintf("%s/%s.txt", out_dir, id_subject))
  
  dat[, n_segment:=NULL]
  #  dat[, uV_filt:=filtruj(bf7, uV), by=.(Subject, Item, word_pos, electrode)] #A different filter for detection of artifacts
  dat[, uV_filt:=uV]
  
  #baseline
  dat2 = dat[dp < 0, .(baseline=mean(uV_filt)), by=.(Subject, Item, word_pos, electrode)]
  dat = merge(dat, dat2, by=c("Subject", "Item", "word_pos", "electrode"))
  dat[, uV_bline:=uV_filt-baseline]
  dat[, baseline:=NULL]
  rm (dat2)
  
  #Look for remaining artifacts that would bias the computation of variance
  dat[, out:=(max(abs(uV_bline)) > 10000), by=.(Subject, Item, word_pos, electrode)]
  
  #cat(sprintf("%s\tSD %d\n", id_subject, dat[electrode %in% c("MiCe", "LMFr", "RMFr"), round(sd(uV_bline))]))
  
  
  artf = dat[, .(diff=max_minmax(uV_bline), mean_uV=mean(uV_bline)), 
             by=.(Subject, Item, word_pos, electrode, out)]
  rm (dat)
  gc()
  return (artf)
}


artf_agr = rbindlist(artf_agr)

artf_agr[, abs_mean_uV:=abs(mean_uV), by=.(Subject, out)]
artf_agr[, s_abs_mean_uV:=scale(abs_mean_uV), by=.(Subject, electrode, out)]
artf_agr[, s_diff:=scale(diff), by=.(Subject, electrode, out)]



artf_agr[, logodds:=3.554876512 + -0.001930831*abs_mean_uV + -0.691452150*s_diff]
artf_agr[, pred:=plogis(logodds)]
artf_agr[out==TRUE, pred:=0]

artf_agr[, potential_in:=ifelse(pred<0.5, 0, 1)]

artf_agr[, mean_pred:=mean(pred), by=.(Subject, Item, word_pos)]
artf_agr[, pred2:=plogis(14.7252 + -20.0935 * mean_pred)]

artf_agr[, my_in:=ifelse(potential_in==0 | pred2>0.5, 0, 1)]

fwrite(artf_agr[, .(Subject, Item, word_pos, electrode, my_in)], file=file_artifacts, sep="\t")
artf_agr[, mean(my_in), by=.(Subject)]
