
#window_len_dp - length of the compared step (so one windows_len_dp gets compared with another one)
#step_size_dp - step size; window_len_dp must be a multiple of step_size_dp
step_detect = function(dt, electrode="HEOG", window_len_dp=20, step_size_dp=10) {
  if (window_len_dp %% step_size_dp != 0) {
    stop("windows_len_dp must be a multiple of step_size_dp")
  }
  
  dt = copy(dt)
  setnames(dt, electrode, "test_el")
  dt[, seg:=rep(1:((nrow(dt))/step_size_dp), each=step_size_dp)]
  dt2 = dt[, .(test_el=mean(test_el)), by=.(seg)]
  
  steps_in_window = window_len_dp / step_size_dp
  
  x = frollmean(dt2[, test_el], steps_in_window)
  
  return (max(frollapply(x, steps_in_window+1, function(pair) {return (abs(pair[1] - pair[steps_in_window+1])) }), na.rm=T))
}

filtruj = function (bf, dane) {
  #2 oscillation cycles padding to avoid filter border artifacts
  dane = c(rev(dane[1:72]), dane, rev(dane[(length(dane)-71):length(dane)]))
  ret = as.double(filtfilt(bf, dane))
  ret = ret[73:(length(ret)-72)]
  return (round(ret))
}

max_minmax = function (dane) {
  max_diff = 0
  for (dx in 101:length(dane)) { #moving window of 400ms
    frame = dane[(dx-100):dx]
    diff = max(frame) - min(frame)
    #print (diff)
    if (diff > max_diff) {
      max_diff = diff
    }
  }
  max_diff
}

uV_to_IC = function (dt_wide) {
  id_subject = dt_wide[1, Subject]
  
  fwd = fread(sprintf("ica/%s.ica.fwd.txt", id_subject))
  fwd[, V1:=NULL]
  fwd_electrodes = colnames(fwd)
  fwd = as.matrix(fwd)
  A = dt_wide[, fwd_electrodes, with=F] #also reorders electrodes to match fwd
  A = as.matrix(A)
  a_fwd = A %*% t(fwd)
  
  id_cols = c("Subject", "exp_ver", "Item", "word_pos", "dp", "n_segment", "noun_pos")
  
  dt2_wide = dt_wide[, id_cols, with=F]
  dt2_wide = cbind(dt2_wide, a_fwd)
  
  return (dt2_wide)
}

get_topodat = function(dat, lme_formula, vars, cores=8) {
  all_electrodes = unique(dat$electrode)
  all_electrodes = all_electrodes[!all_electrodes %in% c("A2", "HEOG", "VEOG", "lhe", "rhe", "LE")]
  
  library(doParallel)
  cl <- makeCluster(cores, outfile="")
  registerDoParallel(cl) 
  ret = foreach (el = all_electrodes, .packages=c("data.table", "lme4"), .noexport=c("eeg"), .verbose=T) %dopar% {
    dat2 = dat[electrode==el]
    m = lmer(lme_formula,
             data=dat2, 
             control=lmerControl(optimizer="bobyqa"), 
             REML=F)
    
    fe = fixef(m)
    se = sqrt(diag(vcov(m))); names(se) = names(fe)
    
    ret = data.table(electrode = el)
    for (var in vars) {
      ret[, (paste0("eff_", var)):=fe[var]]
      ret[, (paste0("se_", var)):=se[var]]
      ret[, (paste0("t_", var)):=fe[var] / se[var]]
    }
    return (ret)
  }
  stopCluster(cl); rm(cl)
  ret = rbindlist(ret)
  
  loc = fread("kdflab polar.txt")
  loc[, radianTheta:=pi/180*theta]
  loc[, x:=radius*sin(radianTheta)]
  loc[, y:=radius*cos(radianTheta)]
  
  ret = merge(ret, loc[, .(electrode, x, y)])
  return (ret)
}

plot_BF = function(dat, xvar, xlab, legend_x_pos) {
  cols = c("#9ecae1", "#6baed6", "#4292c6")
  return (ggplot(dat, aes_string(x=xvar, y="BF")) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(0.33), ymax=log10(3), fill="grey95", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(0.1), ymax=log10(0.33), fill="grey90", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(3), ymax=log10(10), fill="grey90", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(0.033), ymax=log10(0.1), fill="grey85", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(10), ymax=log10(30), fill="grey85", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(0.005), ymax=log10(0.033), fill="grey80", color=NA) + 
            geom_rect(xmin=-100, xmax=100, ymin=log10(30), ymax=log10(200), fill="grey80", color=NA) + 
            geom_point() + geom_line() + 
            scale_y_log10(breaks=c(0.033, 0.1, 0.33, 1, 3, 10, 30), 
                          labels=c("1/30", "1/10", "1/3", "1", "3", "10", "30")) +
            geom_hline(yintercept=1, linetype="dashed") + 
            theme_light(base_size=7) + 
            coord_cartesian(ylim=c(1/70, 70)) + 
            annotate(geom="text", x=legend_x_pos, y=7/10, label="Evidence in favor of H0", size=2) + 
            annotate(geom="text", x=legend_x_pos, y=10/7, label="Evidence in favor of H1", size=2) + 
            xlab(xlab))
}

geom_posterior = function(scale) {
  cols = c("#9ecae1", "#6baed6", "#4292c6")
  return (list(
    geom_vline(xintercept=0, linetype="dashed"),
    ggridges::stat_density_ridges(geom = "density_ridges_gradient",
                                  calc_ecdf = T,
                                  aes(fill=stat(quantile)),
                                  quantiles=c(0.025, 0.33, 0.67, 0.975),
                                  scale=scale),
    scale_fill_manual(values=cols[c(1,2,3,2,1)]),
    guides(fill="none"),
    coord_cartesian(clip='off'),
    scale_x_continuous(expand=c(0,0)),
    scale_y_discrete(NULL, expand=c(0,0)),
    ggridges::theme_ridges(font_size=7, center_axis_labels = T),
    ylab(NULL),
    xlab("Coefficient value"),
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())
  ))
}

geom_topomap = function() {
  return (list(
    geom_topo(interp_limit="head", 
              colour="black",
              chan_size=rel(0.75),
              head_size=rel(0.5)),
    scale_fill_distiller("t-value", 
                         palette = "RdBu", 
                         limits=c(-7, 7), 
                         oob=scales::squish),
    theme_void(base_size=7),
    scale_y_continuous(expand=c(0, 0)),
    theme(plot.margin = margin(0,0,0,12, "points"),
          legend.text=element_text(size=6), 
          legend.title=element_text(size=6),
          legend.key.width=unit(0.2, "cm"),
          legend.key.height = unit(0.50, "cm"),
          legend.margin=margin(0),
          legend.box.margin=margin(-15),
          axis.title.x = element_text())
  ))
}

geom_erp = function() {
  return (list(
    geom_line(),
    scale_y_continuous("Amplitude [uV]", trans="reverse", breaks=seq(-10, 10, 2)),
    scale_x_continuous("Time [ms]", breaks=seq(0, 1600, 400)),
    geom_hline(yintercept=0, alpha=0.5),
    geom_vline(xintercept=0, alpha=0.5),
    theme_light(base_size = 7),
    theme(strip.background = element_blank(),
          strip.text.x = element_blank(),
          legend.position="bottom",
          legend.spacing=unit(0, "cm"),
          legend.box.margin=margin(-5),
          legend.margin=margin(t=0, unit="cm"),
          legend.key.height=unit(0.1, "cm")
    )
  ))
}

add_ant_lat = function(dat) {
  dat[, lat:=factor(str_sub(electrode, 1, 2), levels=c("LL", "LD", "LM", "Mi", "RM", "RD", "RL"))]
  dat[electrode == "VEOG", lat:="LM"]
  dat[electrode == "HEOG", lat:="RL"]
  dat[electrode == "A2", lat:="RL"]
  dat[electrode %in% c("LLPf", "MiPf", "RLPf"), ant:="Pf"]
  dat[electrode %in% c("LMPf", "RMPf"), ant:="PfFr"]
  dat[str_sub(electrode, 3, 4) == "Fr", ant:="Fr"]
  dat[str_sub(electrode, 3, 4) == "Ce", ant:="Ce"]
  dat[electrode %in% c("LLTe", "LDPa", "MiPa", "RDPa", "RLTe"), ant:="Pa"]
  dat[electrode %in% c("LMOc", "RMOc"), ant:="PaOc"]
  dat[electrode %in% c("LLOc", "MiOc", "RLOc"), ant:="Oc"]
  dat[electrode %in% c("HEOG", "VEOG"), ant:="EOG"]
  dat[electrode=="A2", ant:="PaOc"]
  dat[, ant:=factor(ant, levels=c("EOG", "Pf", "PfFr", "Fr", "Ce", "Pa", "PaOc", "Oc"))]
}

geom_erp_fullhead = function() {
  return (list(
    geom_line(),
    facet_grid(ant ~ lat),
    scale_y_continuous("Amplitude [uV]", trans="reverse", breaks=seq(-10, 10, 2)),
    scale_x_continuous("Time [ms]", breaks=seq(0, 1600, 400)),
    geom_hline(yintercept=0, alpha=0.5), 
    geom_vline(xintercept=0, alpha=0.5),
    xlab(NULL),
    theme_light(base_size = 7),
    geom_text(aes(label=electrode), vjust="top", hjust="left", x=20, y=4, size=2, color="black"),
    theme(legend.position="bottom",
          strip.background = element_blank(),
          panel.spacing = unit(1, "points"),
          panel.border = element_blank(),
          strip.text.x = element_blank(),
          strip.text.y = element_blank())
    
  ))
}
