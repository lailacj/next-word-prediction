library(data.table)
library(ggplot2)
library(patchwork)
library(brms)
library(lme4)
library(stringr)
library(signal)
library(eegUtils)
library(doParallel)
source("functions public.R")


#DATA_DIR = "f:/data/2019.05 - sem upd"
DATA_DIR = "."

erp_scale = 8
DPI = 1000
in_dir = paste0(DATA_DIR, "/post-ica segments")
artifacts_file = "artifacts.txt"


#Note: this part of the script requires preprocessed segmented files, which are available in the Harvard Dataverse repository
#To reproduce brms results, go to further sections

files = list.files(in_dir, pattern="*.txt", full.names=T)
eeg = data.table()
for (file in files) {
  a = fread(file, encoding="UTF-8",
            colClasses=list(factor=c(1,2,8), integer=c(3,4,5,6,7,9)))
  eeg = rbind(eeg, a)
}
rm (a)

eeg = eeg[!Subject %in% c("v1B_TTRVZ", "v4_ZTEFN")]
gc()

# Apply artifact rejection
artf = fread(artifacts_file,
             colClasses=list(factor=c(1,2,4), integer=c(3,5)))
eeg = merge(eeg, artf, by=c("Subject", "Item", "word_pos", "electrode"), all.x=T)
eeg = eeg[my_in==1]
eeg[, my_in:=NULL]
rm (artf)
gc()


eeg[, uV:=uV/100] 
eeg[, ms:=as.integer(dp*1000/250)]
eeg[, noun_ms:=as.integer((dp - noun_pos)*1000/250)]
eeg[, noun_pos:=NULL]
eeg[, dp:=NULL]


#Add EOG
eeg_wide = dcast(eeg[electrode %in% c("lhe", "rhe", "LLPf", "LE")], ... ~ electrode, value.var="uV")
eeg_wide[, HEOG:=lhe-rhe/2]
eeg_wide[, VEOG:=LLPf-LE/2]
eeg_wide[, c("lhe", "rhe", "LLPf", "LE"):=NULL]
eeg_wide = melt(eeg_wide, measure.vars=c("HEOG", "VEOG"), variable.name="electrode", value.name="uV")
eeg_wide = eeg_wide[!is.na(uV)]
eeg = rbind(eeg, eeg_wide)
rm (eeg_wide)



#Compute baseline
pre_adj_bline = eeg[ms >= -200 & ms < 0 , .(preadj_bline=mean(uV)), by=.(Subject, Item, electrode, word_pos)]
eeg = merge(eeg, pre_adj_bline, by=c("Subject", "Item", "word_pos", "electrode"))

pre_noun_bline = eeg[noun_ms >= -200 & noun_ms < 0, .(prenoun_bline=mean(uV)), by=.(Subject, Item, electrode, word_pos)]
eeg = merge(eeg, pre_noun_bline, by=c("Subject", "Item", "word_pos", "electrode"))


#Load item info
items = fread("items public.txt")
items[, Item:=as.factor(Item)]
items[, adj_cond:=factor(adj_cond)]
items[, noun_cond:=factor(noun_cond)]



# ----------- descriptive stats  ---------

# Table 1. Cloze probabilities of HiCP and LoCP nouns without an adjective and after pro-HiCP and pro-LoCP adjectives. 

fwrite(items[, .(p_preadj=sprintf("%.2f (%.2f)", mean(p_preadj), sd(p_preadj)),
                 p_postadj=sprintf("%.2f (%.2f)", mean(p_postadj), sd_p_postadj=sd(p_postadj)),
                 constr_preadj=sprintf("%.2f (%.2f)", mean(constr_preadj), sd(constr_preadj)),
                 DKL=sprintf("%.2f (%.2f)", mean(DKL), sd(DKL))), by=.(adj_cond, noun_cond)], 
       file="item stats.txt", sep="\t")




# ------------- general graphs --------------------
if (0) {
  # Figure 1 - distributions of noun CP and how they are affected by adjectives
  items2 = melt(items, measure.vars=c("p_preadj", "p_postadj"))
  items2[adj_cond=="low", adj_cond:="pro-LoCP adj"]
  items2[adj_cond=="top", adj_cond:="pro-HiCP adj"]
  items2[, adj_cond_org:=adj_cond]
  items2[variable=="p_preadj", adj_cond:="no adj"]
  items2[variable=="p_preadj", var_cont:=1]
  items2[variable=="p_postadj", var_cont:=2]
  items2[variable=="p_preadj", variable:="before\nadjective"]
  items2[variable=="p_postadj", variable:="after\nadjective"]
  items2[noun_cond=="top", noun_cond:="HiCP noun"]
  items2[noun_cond=="low", noun_cond:="LoCP noun"]
  items2[, adj_cond:=factor(adj_cond, levels=c("no adj", "pro-HiCP adj", "pro-LoCP adj"))]
  jit = data.table(Item=unique(items2$Item))
  jit[, jitter:=rnorm(nrow(jit), 0, 0.05)]
  set.seed(123)
  jit[, include:=sample(0:1, size=nrow(jit), prob=c(.7, .3), replace=T)]
  jit = jit[include==1]
  items3 = merge(items2, jit)
  items3[, var_cont:=var_cont+jitter]
  ggplot(items3, aes(x=var_cont, y=value, color=adj_cond)) + 
    #  geom_point(alpha=0.3, size=0.5, color=NA, aes(fill=adj_cond)) + 
    geom_line(aes(group=interaction(Item, adj_cond_org), color=adj_cond_org), alpha=0.3) + 
    facet_wrap ( ~ noun_cond) + 
    xlab(NULL) + ylab("Cloze probability") + 
    scale_x_continuous(breaks=c(1, 2), labels=c("before\nadjective", "after\nadjective")) + 
    scale_color_manual(NULL, values=c("no adj"="black", "pro-HiCP adj"="#1b9e77", "pro-LoCP adj"="#7570b3")) + 
    scale_fill_manual(NULL, values=c("no adj"="black", "pro-HiCP adj"="#1b9e77", "pro-LoCP adj"="#7570b3")) + 
    theme_minimal(base_size=8) + 
    theme(legend.position="bottom")
  ggsave("Figure 1.png", width=9, height=7, unit="cm", dpi=DPI)



  items = fread("items public.txt")
  items[, Item:=as.factor(Item)]
  items[, adj_cond:=factor(adj_cond)]
  items[, noun_cond:=factor(noun_cond)]
  
  
  #Figure 2
  items[, noun_cond_full := ifelse(noun_cond=="top", "HiCP noun", "LoCP noun")]
  items[, adj_cond_full := ifelse(adj_cond=="top", "pro-HiCP adj          ", "pro-LoCP adj")]
  items[, p_update := p_postadj - p_preadj]
  
  #Figure 2a - distribution of noun preadj CP
  fig2a = ggplot(items, aes(x=p_preadj)) +
    geom_density(alpha=0.5, fill="black", size=0.3) +
    facet_wrap (~ noun_cond_full) +
    theme_light(base_size=7) +
    scale_x_continuous("Noun CP (pre-adjectival)", breaks=seq(0, 1, 0.25), labels=c("0", ".25", ".5", ".75", "1")) + 
    ggtitle("Pre-adjectical target noun CP")

  #Figure 2b - distribution of CP update
  fig2b = ggplot(items, aes(x=p_update, fill=adj_cond_full)) + 
    facet_wrap (~ noun_cond_full) + 
    geom_density(alpha=0.5, size=0.3) + 
    scale_fill_manual(NULL, values=c("pro-HiCP adj          "="#1b9e77", "pro-LoCP adj"="#7570b3")) + 
    scale_x_continuous (breaks=seq(-1, 1, 0.25), labels=c("-1", "-.75", "-.5", "-.25", "0", ".25", ".5", ".75", "1")) + 
    xlab("Δ CP (Noun CP update)") + 
    geom_vline(xintercept=0, linetype="dashed", color="gray30", size=0.3) + 
    theme_light(base_size=7) + 
    theme(legend.key.size=unit(9, "points"),
          legend.position="bottom") + 
    ggtitle("Adjective-driven update to target noun CP")

  #Figure 2c - distribution of DKL
  fig2c = ggplot(items, aes(x=DKL, fill=adj_cond_full)) + 
    geom_density(alpha=0.5, size=0.3) + 
    scale_fill_manual(NULL, values=c("pro-HiCP adj          "="#1b9e77", "pro-LoCP adj"="#7570b3")) + 
    xlab(expression(D["KL"])) + 
    theme_light(base_size=7) + 
    theme(legend.key.size=unit(9, "points"),
          legend.position="bottom") + 
    ggtitle(expression(D["KL"]))
  
  
  lo = "
AABBBBC
##DDDDD
"
  fig2a + fig2b + fig2c + guide_area() + 
    plot_layout(guides='collect', design=lo, heights = c(30,1))
  
  ggsave("Figure 2 R1.png", height=5, width=19, unit="cm", dpi=DPI)
  

  #Figure 3
  ggplot(items, aes(x=p_update, y=p_preadj)) + 
    geom_jitter(alpha=0.4, shape=16, size=0.7) + 
    scale_x_continuous("Δ CP (Noun CP update)", breaks=seq(-1, 1, 0.2)) +
    scale_y_continuous("Noun CP before ADJ", breaks=seq(0, 1, 0.2)) +
    geom_vline(xintercept=0, linetype="dashed", size=0.5) +
    theme_light(base_size=7)
  ggsave(file="Figure 3.png", width=9, height=5, unit="cm", dpi=DPI)
  

  #Not shown in the paper - the distribution of DKL for pro-HiCP and pro-LoCP adjectives
  ggplot(items, aes(x=DKL, fill=adj_cond_full)) + geom_density(alpha=0.6) +
    theme_bw(base_size=12) + 
    scale_fill_manual(NULL, values=c("pro-HiCP adj"="#1b9e77", "pro-LoCP adj"="#7570b3"))
  ggsave("adj DKL distr.png", height=8, width=13, unit="cm", dpi=DPI)



  #GGA plots at the noun - 4 "conditions"
  eeg = eeg[items, on=c("Item", "exp_ver"), `:=`(adj_cond=i.adj_cond, noun_cond=i.noun_cond)]
  dat = eeg[!is.na(adj_cond) & electrode %in% c("MiPf", "MiCe", "MiPa", "MiOc")]
  bf15 = butter(2, 15/125, "low") #the real cutoff frequency will be higher, approx 3/2 times the stated freq
  dat[, uV_filt:=filtruj(bf15, uV*1000)/1000, by=.(Subject, Item, word_pos, electrode)]
  dat = dat[, 
            .(uV=mean(uV_filt), prenoun_bline=mean(prenoun_bline)), 
            by=.(electrode, noun_ms, adj_cond, noun_cond)] #GGA
  dat[, uV_bline:=uV-prenoun_bline]
  dat[, noun_cond:=ifelse(noun_cond=="top", "HiCP", "LoCP")]
  dat[, adj_cond:=ifelse(adj_cond=="top", "pro-HiCP", "pro-LoCP")]
  dat[, cond:=paste0("(", adj_cond, " adj, ", noun_cond, " noun)")]
  dat[, cond:=fcase(
    cond == "(pro-HiCP adj, HiCP noun)", cond=paste0("front door\n", cond),
    cond == "(pro-HiCP adj, LoCP noun)", cond=paste0("front window\n", cond),
    cond == "(pro-LoCP adj, HiCP noun)", cond=paste0("frosty door\n", cond),
    cond == "(pro-LoCP adj, LoCP noun)", cond=paste0("frosty window\n", cond)
  )]
  dat = dat[noun_ms > -100 & noun_ms < 890]
  
  rng = dat[, max(uV_bline) - min(uV_bline), by=.(electrode)][, max(V1)]
  mids = dat[, max(uV_bline)/2  + min(uV_bline)/2 , by=.(electrode)]
  mids[, `:=`(ymin=V1-rng/2, ymax=V1+rng/2)]
  mids = melt(mids, id.var=c("electrode"), measure.vars=c("ymin", "ymax"))
  mids[, cond:=dat[, unique(cond)][1]]
  
  #Figure 4
  ggplot(dat, 
         aes(x=noun_ms, y=uV_bline, linetype=cond, color=cond)) +
    geom_rect(xmin=300, xmax=500, ymin=-100, ymax=100, fill="grey95", color=NA) +
    geom_line() +
    geom_point(color=NA, data=mids, aes(x=0, y=value)) + 
    facet_grid(factor(electrode, levels=c("MiPf", "MiCe", "MiPa", "MiOc")) ~., scale="free_y") + 
    scale_x_continuous("Time [ms]", breaks=seq(0, 1600, 400)) +
    scale_y_continuous("Amplitude [uV]", trans="reverse", breaks=seq(-10, 10, 2)) + 
    geom_hline(yintercept=0, alpha=0.5) + geom_vline(xintercept=0, alpha=0.5) +
    xlab(NULL) + 
    geom_text(x=5, y=1, hjust=0, aes(label=electrode), size=2, color="black", fontface="plain") + 
    theme_light(base_size = 7) + 
    scale_linetype_manual("At night the old woman locked the ...", values=c("solid", "longdash", "solid", "longdash")) + 
    scale_color_manual("At night the old woman locked the ...", values=c("#1b9e77", "#1b9e77", "#7570b3", "#7570b3")) + 
    guides(color=guide_legend(title.position="top", nrow=2),
           linetype=guide_legend(title.position="top", nrow=2)) + 
    theme(legend.position="bottom",
          #legend.spacing = unit(20, "mm"),
          legend.margin = margin(0,0,0,0),
          legend.key.width = unit(16, "points"),
          legend.key.height = unit(15, "points"),
          panel.spacing = unit(4, "points"),
          panel.border = element_rect(color = "transparent", fill = "transparent"),
          strip.background = element_blank(),
          strip.text.x = element_blank(),
          panel.grid = element_line(colour = "grey77"))
  ggsave(file=sprintf("Figure 4 R2.png"), dpi=DPI, height=16, width=7, unit="cm")

  
  #Figure S2
  dat = eeg[!is.na(adj_cond) & !electrode %in% c("LE", "lhe", "rhe"), .(uV=mean(uV), prenoun_bline=mean(prenoun_bline)), by=.(electrode, noun_ms, adj_cond, noun_cond)] #GGA
  dat[, uV_bline:=uV-prenoun_bline]
  add_ant_lat(dat)
  ggplot(dat[noun_ms > -100 & noun_ms < 890], aes(x=noun_ms, y=uV_bline, linetype=noun_cond, color=adj_cond)) +
    geom_erp_fullhead()
    ggtitle("Noun") + 
    scale_linetype_discrete("Noun\ncondition") + 
    scale_color_manual("Adjective\ncondition", values=c("pro-HiCP"="#1b9e77", "pro-LoCP"="#7570b3"))
  ggsave("Figure S2.png", height=16, width=24, unit="cm", dpi=DPI)
  
  
  #GGA at adjectives (DKL, logp)
  eeg = eeg[items, on=c("Item", "exp_ver"), `:=`(gpt_logp_adj=i.gpt_logp_adj, DKL=i.DKL)]
  
  eeg[, DKL:=factor(ifelse(DKL<median(DKL, na.rm=T), "low", "high"))]
  eeg[, gpt_logp_adj:=factor(ifelse(gpt_logp_adj<median(gpt_logp_adj, na.rm=T), "low", "high"))]
  
  #Figure S3 (DKL)
  dat = eeg[!is.na(DKL) & !electrode %in% c("LE", "lhe", "rhe"), .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(electrode, ms, DKL)] #GGA
  dat[, uV_bline:=uV-preadj_bline]
  add_ant_lat(dat)
  ggplot(dat[ms > -100 & ms < 890], aes(x=ms, y=uV_bline, linetype=DKL)) +
    geom_erp_fullhead() + 
    ggtitle("Adjective") + 
    scale_linetype_discrete(name=bquote(D[KL]))
  ggsave("Figure S3.png", height=16, width=24, unit="cm", dpi=DPI)
  

  #Figure S4 (ADJ pro-HiCP vs pro-LoCP)  
  dat = eeg[!is.na(adj_cond) & !electrode %in% c("LE", "lhe", "rhe"), .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(electrode, ms, adj_cond)] #GGA
  dat[, uV_bline:=uV-preadj_bline]
  dat[, adj_cond:=ifelse(adj_cond=="top", "pro-HiCP", "pro-LoCP")]
  add_ant_lat(dat)
  ggplot(dat[ms > -100 & ms < 890], aes(x=ms, y=uV_bline, color=adj_cond)) +
    geom_erp_fullhead() + 
    ggtitle("Adjective") + 
    scale_color_manual("Adjective condition", values=c("pro-HiCP"="#1b9e77", "pro-LoCP"="#7570b3"))
  ggsave("Figure S4.png", height=16, width=24, unit="cm", dpi=DPI)
  
  

# --------- prepare files for stats -------------------


prepare_for_stats = function(dt) {
  dt = merge(dt, items, by=c("Item", "exp_ver"), all.x=T)

  dt = dt[!is.na(adj_cond)] #fillers out

  dt[, c_trial:=(n_segment-mean(n_segment))/100]
  dt[, c_preadj_bline:=(preadj_bline-mean(preadj_bline))]
  dt[, c_prenoun_bline:=(prenoun_bline-mean(prenoun_bline))]
  dt[, c_DKL:=scale(DKL, scale=F)]
  dt[, c_p_preadj:=scale(p_preadj, scale=F)]
  dt[, c_p_postadj:=scale(p_postadj, scale=F)]
  dt[, c_constr_preadj:=scale(constr_preadj, scale=F)]
  dt[, p_update := p_postadj - p_preadj]
  dt[, c_p_update := scale(p_update, scale=F)]
  dt[, c_adj_gpt_logp:=scale(adj_gpt_logp, scale=F)]
  dt[, adj_logfreq:=log(adj_freq+1)]
  dt[, noun_logfreq:=log(noun_freq+1)]
  dt[, c_adj_logfreq:=scale(adj_logfreq, scale=F)]
  dt[, c_noun_logfreq:=scale(noun_logfreq, scale=F)]
  dt[, c_adj_pld20:=scale(adj_pld20, scale=F)]
  dt[, c_noun_pld20:=scale(noun_pld20, scale=F)]
  
  return (dt)
}


dat.noun.n400 = eeg[electrode %in% c("MiCe", "MiPa", "LMCe", "RMCe") & noun_ms >= 300 & noun_ms <= 500, 
                    .(uV=mean(uV), preadj_bline=mean(preadj_bline), prenoun_bline=mean(prenoun_bline)), 
                    by=.(Subject, Item, exp_ver, n_segment)]
dat.noun.n400 = prepare_for_stats(dat.noun.n400)
dat.noun.n400[, `:=`(freq=noun_freq, pld20=noun_pld20, logfreq=noun_logfreq, c_logfreq=c_noun_logfreq, c_pld20=c_noun_pld20)]


dat.adj.n400 = eeg[electrode %in% c("MiCe", "MiPa", "LMCe", "RMCe") & ms >= 300 & ms <= 500, 
                   .(uV=mean(uV), preadj_bline=mean(preadj_bline), prenoun_bline=mean(prenoun_bline)),
                   by=.(Subject, Item, exp_ver, n_segment)]
dat.adj.n400 = prepare_for_stats(dat.adj.n400)
dat.adj.n400[, `:=`(freq=adj_freq, pld20=adj_pld20, logfreq=adj_logfreq, c_logfreq=c_adj_logfreq, c_pld20=c_adj_pld20)]


fwrite(dat.noun.n400, "data noun n400.txt")
fwrite(dat.adj.n400, "data adj n400.txt")



# ---- Noun - analysis ----
dat.noun.n400[, Item:=factor(Item)]
dat.noun.n400[, Subject:=factor(Subject)]

dat.noun.n400[, s_concr:=scale(concr_noun)]
dat.noun.n400[, s_pld20:=scale(pld20_noun)]
dat.noun.n400[, s_logfreq:=scale(log(freq_noun+1))]
dat.noun.n400[, s_ctx_n_words:=scale(ctx_n_words)]

priors.noun = c(prior(normal( 0, 3), class = Intercept),
                prior(normal( 0.5, 0.3), class = b, coef="c_preadj_bline"),
                prior(normal( 0.25, 0.15), class = b, coef="c_prenoun_bline"),
                prior(normal( 3, 2), class = b, coef="c_p_preadj"),
                prior(normal( 3, 2), class = b, coef="c_p_update2"),
                prior(normal( 0, 1), class = b, coef="s_concr"),
                prior(normal( 0, 1), class = b, coef="s_pld20"),
                prior(normal( 0, 1), class = b, coef="s_logfreq"),
                prior(normal( 0, 1), class = b, coef="s_ctx_n_words"),
                prior(normal(10, 3), class = sigma),
                prior(normal( 0, 2), class = sd)
)


# Baseline model (coef=1) and other models with varying neg/pos updating ratio
warmup = 1500
iter=28500
cores=14
dir = "noun/"
prefix = "noun_"

for (coef_neg in c(0.5, 0, 0.1, 0.2, 0.33, 0.66, 0.8, 1)) {
  dat.noun.n400[, p_update2:=p_update]
  dat.noun.n400[p_update<0, p_update2:=p_update2*coef_neg] #neg-to-pos updating ratio
  dat.noun.n400[, c_p_update2:=p_update2 - mean(p_update2)]
  
  bm_noun = brm(uV ~ c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + 
                  (c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 || Subject) +
                  (c_preadj_bline + c_prenoun_bline + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 || Item)
                , prior = priors.noun,
                warmup  = warmup,
                iter    = iter,
                cores   = cores,
                chains  = cores,
                control = list(adapt_delta = 0.9),
                save_pars = save_pars(all = TRUE),
                data    = dat.noun.n400,
                file    = NULL
  )
  
  cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, coef_neg))
  cat(paste(capture.output(summary(bm_noun)), collapse="\n"))
  cat("\n\n")
  
  if (coef_neg == 0.66) {
    post = data.table(posterior_samples(bm_noun, pars=c("b_c_p_update2", "b_c_p_preadj")))
    saveRDS(post, paste0(dir, "posterior_", prefix, coef_neg, ".rds"))
    rm (post)
    
    dat = unique(dat.noun.n400[, .(c_p_update2=0, c_preadj_bline=0, c_prenoun_bline=0, p_preadj, c_p_preadj, s_logfreq=0, s_ctx_n_words=0, s_pld20=0, s_concr=0)])
    preds1 = as.data.table(fitted(bm_noun, newdata=dat, re_formula=NA))
    preds1 = cbind(dat[, .(p_preadj)], preds1)
    preds1[, `:=`(var="p_preadj",
                  p_update=0)]
    
    dat = unique(dat.noun.n400[, .(c_p_update2, p_update, c_preadj_bline=0, c_prenoun_bline=0, c_p_preadj=0, s_logfreq=0, s_ctx_n_words=0, s_pld20=0, s_concr=0)])
    preds2 = as.data.table(fitted(bm_noun, newdata=dat, re_formula=NA))
    preds2 = cbind(dat[, .(p_update)], preds2)
    preds2[, `:=`(var="p_update2",
                  p_preadj=0)]
    
    preds = rbind(preds1, preds2)
    saveRDS(preds, paste0(dir, "preds_", prefix, coef_neg, ".rds"))
  }
  
  marg = bridge_sampler(bm_noun, silent = TRUE, cores=8)
  saveRDS(marg, paste0(dir, "marg_", prefix, coef_neg, ".rds"))
  rm (bm_noun)
  gc()
}

marg0 = readRDS(paste0(dir, "marg_", prefix, 0, ".rds"))
ret = data.table()
for (coef_neg in c(0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 1)) {
  marg = readRDS(paste0(dir, "marg_", prefix, coef_neg, ".rds"))
  BF = bayes_factor(marg, marg0)
  ret = rbind(ret, data.table(coef=coef_neg, BF=BF[[1]]))
}
fwrite(ret, paste0(dir, "BF_", prefix, ".txt"), sep="\t")



#Prepare data for the scalp map (this part requires access to unaggregated data in variable `eeg`)
dat = eeg[!electrode %in% c("LE", "lhe", "rhe", "A2", "HEOG", "VEOG") & noun_ms > 300 & noun_ms <= 500,
          .(uV=mean(uV), preadj_bline=mean(preadj_bline), prenoun_bline=mean(prenoun_bline)), 
          by=.(Subject, Item, exp_ver, electrode)]
dat = merge(dat, items, by=c("Item", "exp_ver"))
dat[, `:=`(Subject=factor(Subject),
           Item=factor(Item),
           s_logfreq=scale(log(freq_noun+1)),
           s_pld20=scale(pld20_noun),
           s_ctx_n_words=scale(ctx_n_words),
           s_concr=scale(concr_noun),
           c_preadj_bline=preadj_bline-mean(preadj_bline),
           c_prenoun_bline=prenoun_bline-mean(prenoun_bline),
           p_update=p_postadj-p_preadj)]
dat[, p_update2:=p_update]
dat[p_update<0, p_update2:=p_update2*0.66] #here, we assume neg-to-pos updating ratio = 0.66
dat[, `:=`(c_p_preadj=p_preadj-mean(p_preadj),
           c_p_update2=p_update2-mean(p_update2))]

vars = c("c_p_update2", "c_p_preadj")

#the same formula as for the Bayesian model
formula_noun = as.formula(uV ~ c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + 
  (c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 || Subject) +
  (c_preadj_bline + c_prenoun_bline + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 || Item))

ret = get_topodat(dat, formula_noun, vars, cores=13)
fwrite(ret, "noun/topo noun preadj update.txt", sep="\t")





# ---- Noun - plot (Figure 5) ----

# Figure 5 left (posterior)
x = readRDS("noun/posterior_noun_0.66.rds")
colnames(x) = c("CP update", "Pre-adjectival CP")
x = melt(x, measure.vars=colnames(x))
fig5L = ggplot(x, aes(x=value, y=variable) ) + 
  geom_posterior(scale=0.9) + 
  annotate(geom="text", hjust=0, x=0.6, y=2.8, label="pre-adjectival CP", size=2) + 
  annotate(geom="text", hjust=0, x=0.6, y=1.8, label="CP update", size=2)


# Figure 5 middle-left
preds = readRDS("noun/preds_noun_priors_0.66.rds")
fig5ML_top = ggplot(preds[var=="p_preadj"], aes(x=p_preadj, y=Estimate)) + 
  scale_x_continuous("Pre-adjectival CP", breaks=seq(-1, 1, 0.5))
fig5ML_bottom = ggplot(preds[var=="p_update2"], aes(x=p_update, y=Estimate)) + 
  scale_x_continuous("Δ CP (Noun CP update)", breaks=seq(-1, 1, 0.5)) + 
  geom_hline(yintercept=0, linetype="dashed", size=0.1) +
  geom_vline(xintercept=0, linetype="dashed", size=0.1)

for (varname in c("fig5ML_top", "fig5ML_bottom")) {
  assign(varname, get(varname) + 
           geom_line() + 
           geom_ribbon(aes(ymin=Q2.5, ymax=Q97.5), alpha=0.2, color=NA) + 
           scale_y_continuous("Amplitude (µV)", trans="reverse", limits=c(4, -3.5)) + 
           theme_light(base_size=7)
  )
}


#Figure 5 middle-right
x = fread("noun/BF_noun_.txt")
fig5MR = plot_BF(x, "coef", "neg / pos updating coef", 0.55)

#Figure 5 right (topomaps)
topodat = fread("noun/topo noun preadj update.txt")

fig5R_top = ggplot(topodat, aes(x=x*150, y=y*150, fill=t_s_p_preadj, z=t_s_p_preadj)) + 
  geom_topomap() + 
  xlab("Pre-adjectival CP  300-500 ms")
fig5R_bottom = ggplot(topodat, aes(x=x*150, y=y*150, fill=t_s_p_update2, z=t_s_p_update2)) + 
  geom_topomap() + 
  xlab("CP update   300-500 ms")



design = "
AAAAAAAAABBBBB##############EEEEEEEE
AAAAAAAAACCCCCCCCCCDDDDDDDDDFFFFFFFF
"

#4.5 / 5 / 4.5 / 4
fig5L + fig5ML_top + fig5ML_bottom + fig5MR + 
  fig5R_top + fig5R_bottom + 
  plot_layout(design=design, guides='collect')
ggsave("Figure 5 R1.png", width=18, height=8, unit="cm", dpi=DPI)





# ---- Noun - compare model with CP vs. CP divided into p_preadj and p_update ----
priors.noun = c(prior(normal( 0, 3), class = Intercept),
                prior(normal( 0.5, 0.3), class = b, coef="c_preadj_bline"),
                prior(normal( 0.25, 0.15), class = b, coef="c_prenoun_bline"),
                prior(normal( 3, 2), class = b, coef="c_p_preadj"),
                prior(normal( 3, 2), class = b, coef="c_p_update2"),
                prior(normal( 0, 1), class = b, coef="s_concr"),
                prior(normal( 0, 1), class = b, coef="s_pld20"),
                prior(normal( 0, 1), class = b, coef="s_logfreq"),
                prior(normal( 0, 1), class = b, coef="s_ctx_n_words"),
                prior(normal(10, 3), class = sigma),
                prior(normal( 0, 2), class = sd)
)


coef_neg = 0.66
dat.noun.n400[, p_update2:=p_update]
dat.noun.n400[p_update<0, p_update2:=p_update2*coef_neg]
dat.noun.n400[, c_p_update2:=p_update2 - mean(p_update2)]

warmup = 1500
iter=28500
cores=14
dir = "noun/"
prefix = "noun_comp_"

bm_noun = brm(uV ~ c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + 
                (c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + c_p_postadj || Subject) +
                (c_preadj_bline + c_prenoun_bline + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + c_p_postadj || Item) 
              , prior = priors.noun,
              warmup  = warmup,
              iter    = iter,
              cores   = cores,
              chains  = cores,
              control = list(adapt_delta = 0.9),
              save_pars = save_pars(all = TRUE),
              data    = dat.noun.n400, 
              file    = NULL
)

marg = bridge_sampler(bm_noun, silent = TRUE, cores=8)
saveRDS(marg, paste0(dir, "marg_", prefix, coef_neg, ".rds"))
rm (bm_noun)
gc()


priors.noun2 = c(priors.noun[c(-4, -5),], #remove priors for p_preadj & p_update2
                 prior(normal( 3, 2), class = b, coef="c_p_postadj")
)

bm_noun2 = brm(uV ~ c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_postadj + 
                 (c_preadj_bline + c_prenoun_bline + s_ctx_n_words + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + c_p_postadj || Subject) +
                 (c_preadj_bline + c_prenoun_bline + s_concr + s_pld20 + s_logfreq + c_p_preadj + c_p_update2 + c_p_postadj || Item) 
               , prior = priors.noun2,
               warmup  = warmup,
               iter    = iter,
               cores   = cores,
               chains  = cores,
               control = list(adapt_delta = 0.9),
               save_pars = save_pars(all = TRUE),
               data    = dat.noun.n400, 
               file    = NULL)

marg2 = bridge_sampler(bm_noun2, silent = TRUE, cores=12)
saveRDS(marg2, paste0(dir, "marg_", prefix, "postadj.rds"))
rm (bm_noun2)
gc()

marg = readRDS(paste0(dir, "marg_", prefix, coef_neg, ".rds"))
marg2 = readRDS(paste0(dir, "marg_", prefix, "postadj.rds"))
BF = bayes_factor(marg, marg2)


# ---- Adj - DKL effect - analysis ----
dat.adj.n400 = fread("data adj n400.txt")
dat.adj.n400[, Item:=factor(Item)]
dat.adj.n400[, Subject:=factor(Subject)]

dat.adj.n400[, s_pld20:=scale(pld20_adj)]
dat.adj.n400[, s_logfreq:=scale(log(freq_adj+1))]
dat.adj.n400[, s_ctx_n_words:=scale(ctx_n_words)]

priors.adj = c(prior(normal( 0, 3), class = Intercept),
               prior(normal( 0.5, 0.3), class = b, coef="c_preadj_bline"),
               prior(normal( 0, 1), class = b, coef="s_pld20"),
               prior(normal( 0, 1), class = b, coef="s_logfreq"),
               prior(normal( 0, 1), class = b, coef="s_ctx_n_words"),
               prior(normal(10, 3), class = sigma),
               prior(normal( 0, 2), class = sd))

warmup = 1500
iter = 28500
cores = 14
dir = "adj_DKL/"
prefix = "adj_dkl_"

if (0) {
  bm0 = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq +  
              (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_DKL || Subject) + 
              (c_preadj_bline + s_pld20 + s_logfreq + c_DKL || Item)
            ,
            prior   = priors.adj,
            warmup  = warmup,
            iter    = iter,
            cores   = cores,
            chains  = cores,
            control = list(adapt_delta = 0.9),
            save_pars = save_pars(all = TRUE),
            data    = dat.adj.n400,
            file    = NULL
  )
  
  cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, 0))
  cat(paste(capture.output(summary(bm0)), collapse="\n"))
  cat("\n\n")
  
  marg = bridge_sampler(bm0, silent = TRUE, cores=8)
  saveRDS(marg, paste0(dir, "marg_", prefix, "null.rds"))
  rm (bm0); gc()
  
}

#sensitivity analysis
for (dkl_sd in c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)) {
  stanvars = stanvar(dkl_sd, name="dkl_sd")
  bm = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_DKL + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_DKL || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_DKL || Item)
           ,
           prior   = c(priors.adj, prior(normal( 0, dkl_sd), class = b, coef="c_DKL")),
           warmup  = warmup,
           iter    = iter,
           cores   = cores,
           chains  = cores,
           control = list(adapt_delta = 0.9),
           save_pars = save_pars(all = TRUE),
           data    = dat.adj.n400,
           stanvars=stanvars,
           file    = NULL
  )
  
  if (dkl_sd == 0.75) {
    post = data.table(posterior_samples(bm, pars=c("b_c_DKL")))
    saveRDS(post, paste0(dir, "posterior_", prefix, dkl_sd, ".rds"))
    rm (post)
  }
  
  cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, dkl_sd))
  cat(paste(capture.output(summary(bm)), collapse="\n"))
  cat("\n\n")
  
  marg = bridge_sampler(bm, silent = TRUE, cores=8)
  saveRDS(marg, paste0(dir, "marg_", prefix, dkl_sd, ".rds"))
  rm (bm)
  gc()
}

ret = data.table()
marg0 = readRDS(paste0(dir, "marg_", prefix, "null.rds"))
for (ef in c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)) {
  filename = paste0(dir, "marg_", prefix,  ef, ".rds")
  if (!file.exists(filename)) next
  marg = readRDS(filename)
  BF = bayes_factor(marg, marg0)[[1]]
  ret = rbind(ret, data.table(sd=ef, BF=BF))
}
fwrite(ret, paste0(dir, "BF_", prefix, ".txt"), sep="\t")


#Prepare data for the scalp map
dat = eeg[!electrode %in% c("LE", "lhe", "rhe", "A2", "HEOG", "VEOG") & ms > 300 & ms <= 500, 
          .(uV=mean(uV), preadj_bline=mean(preadj_bline)), 
          by=.(Item, Subject, exp_ver, electrode)] #GGA
rm (eeg); gc()
dat = merge(dat, items, by=c("Item", "exp_ver"))

dat[, `:=`(Subject=factor(Subject),
           Item=factor(Item),
           s_logfreq=scale(log(freq_adj+1)),
           s_pld20=scale(pld20_adj),
           s_ctx_n_words=scale(ctx_n_words),
           c_preadj_bline=preadj_bline-mean(preadj_bline),
           s_DKL=scale(DKL))]

formula_adj_dkl = as.formula(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_DKL + 
                               (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_DKL || Subject) + 
                               (c_preadj_bline + s_pld20 + s_logfreq + s_DKL || Item))
vars = c("s_DKL")
ret = get_topodat(dat, formula_adj_dkl, vars, cores=13)
fwrite(ret, "adj_DKL/topo adj_dkl.txt", sep="\t")

# ---- Adj - DKL effect - plot (Figure 6) ----

#Figure 6 left (DKL)
eeg = eeg[items, on=c("Item", "exp_ver"), `:=`(DKL=i.DKL)]
eeg[, DKL.f:=factor(ifelse(DKL<median(DKL, na.rm=T), "low", "high"))]

dat = eeg[!is.na(DKL.f) & electrode=="MiPa", .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(electrode, ms, DKL.f)] #GGA
dat[, uV_bline:=uV-preadj_bline]

fig6L = ggplot(dat[ms > -100 & ms < 890], aes(x=ms, y=uV_bline, linetype=DKL.f)) +
  geom_erp() +  
  scale_linetype_discrete(name=bquote(D[KL])) + 
  coord_cartesian(ylim=c(5.1, 5.1-erp_scale))

#Fig 6 middle-left
x = readRDS("adj_dkl/posterior_adj_dkl_0.75.rds")
fig6ML = ggplot(x, aes(x=b_c_DKL, y="DKL") ) + 
  geom_posterior(scale=1.2) + 
  annotate(geom="label", x=-0.7, y=2.5, label=expression(D[KL]), size=2, label.size=0, label.padding = unit(0.1, "lines"))


#Figure 6 middle-right
BF = fread("adj_dkl/BF_adj_dkl_.txt")
fig6MR = plot_BF(BF, "sd", expression(D["KL"]*"  Normal prior SD"), legend_x_pos=1.47)


#Figure 6 right
topodat = fread("adj_DKL/topo adj_dkl.txt")
fig6R = ggplot(topodat, aes(x=x*150, y=y*150, fill=t_s_DKL, z=t_s_DKL)) + 
  geom_topomap() + 
  xlab(expression(D[KL]*"   300-500 ms"))


fig6L + fig6ML + fig6MR + fig6R + 
  plot_layout(design="AAAAAAAAABBBBBBBBBBCCCCCCCCCDDDDDDDD") #4.5 / 5 / 4.5 / 4
ggsave("Figure 6 R1.png", width=18, height=5, unit="cm", dpi=DPI)




# ---- Adj - log(gpt2.p) effect - analysis ----
dat.adj.n400 = fread("data adj n400.txt")
dat.adj.n400[, Item:=factor(Item)]
dat.adj.n400[, Subject:=factor(Subject)]

dat.adj.n400[, s_pld20:=scale(pld20_adj)]
dat.adj.n400[, s_logfreq:=scale(log(freq_adj+1))]
dat.adj.n400[, s_ctx_n_words:=scale(ctx_n_words)]

priors.adj = c(prior(normal( 0, 3), class = Intercept),
               prior(normal( 0.5, 0.3), class = b, coef="c_preadj_bline"),
               prior(normal( 0, 1), class = b, coef="s_pld20"),
               prior(normal( 0, 1), class = b, coef="s_logfreq"),
               prior(normal( 0, 1), class = b, coef="s_ctx_n_words"),
               prior(normal(10, 3), class = sigma),
               prior(normal( 0, 2), class = sd))

warmup = 1500
iter = 28500
cores = 14
dir = "adj_logp/"
prefix = "adj_logp_"

bm0 = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + 
            (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_gpt_logp_adj || Subject) + 
            (c_preadj_bline + s_pld20 + s_logfreq + c_gpt_logp_adj || Item)
          ,
          prior   = priors.adj,
          warmup  = warmup,
          iter    = iter,
          cores   = cores,
          chains  = cores,
          control = list(adapt_delta = 0.9),
          save_pars = save_pars(all = TRUE),
          data    = dat.adj.n400,
          file    = NULL
)
cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, 0))
cat(paste(capture.output(summary(bm0)), collapse="\n"))
cat("\n\n")

marg = bridge_sampler(bm0, silent = TRUE, cores=8)
saveRDS(marg, paste0(dir, "marg_", prefix, "null.rds"))
rm (bm0); gc()

#sensitivity analysis
for (logp_sd in c(0.03, 0.07, 0.09, 0.11, 0.15, 0.2, 0.3, 0.4, 0.6)) {
  stanvars = stanvar(logp_sd, name="logp_sd")
  bm = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_gpt_logp_adj + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_gpt_logp_adj || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_gpt_logp_adj || Item)
           ,
           prior   = c(priors.adj, prior(normal( 0, logp_sd), class = b, coef="c_gpt_logp_adj")),
           warmup  = warmup,
           iter    = iter,
           cores   = cores,
           chains  = cores,
           control = list(adapt_delta = 0.9),
           save_pars = save_pars(all = TRUE),
           data    = dat.adj.n400,
           stanvars=stanvars,
           file    = NULL
  )
  
  if (logp_sd == 0.11) {
    post = data.table(posterior_samples(bm, pars=c("b_c_gpt_logp_adj")))
    saveRDS(post, paste0(dir, "posterior_", prefix, logp_sd, ".rds"))
    rm (post)
  }
  
  cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, logp_sd))
  cat(paste(capture.output(summary(bm)), collapse="\n"))
  cat("\n\n")
  
  marg = bridge_sampler(bm, silent = TRUE, cores=8)
  saveRDS(marg, paste0(dir, "marg_", prefix, logp_sd, ".rds"))
  rm (bm)
  gc()
}

marg0 = readRDS(paste0(dir, "marg_", prefix, "null.rds"))
ret = data.table()
for (ef in c(0.03, 0.07, 0.09, 0.11, 0.15, 0.2, 0.3, 0.4, 0.6)) {
  marg = readRDS(paste0(dir, "marg_", prefix, ef, ".rds"))
  BF = bayes_factor(marg, marg0)
  ret = rbind(ret, data.table(sd=ef, BF=BF[[1]]))
}
fwrite(ret, paste0(dir, "BF_", prefix, ".txt"), sep="\t")


#Prepare data for the scalp map
dat = eeg[!electrode %in% c("LE", "lhe", "rhe", "A2", "HEOG", "VEOG") & ms > 300 & ms <= 500, 
          .(uV=mean(uV), preadj_bline=mean(preadj_bline)), 
          by=.(Item, Subject, exp_ver, electrode)]
dat = merge(dat, items, by=c("Item", "exp_ver"))

dat[, `:=`(Subject=factor(Subject),
           Item=factor(Item),
           s_logfreq=scale(log(freq_adj+1)),
           s_pld20=scale(pld20_adj),
           s_ctx_n_words=scale(ctx_n_words),
           c_preadj_bline=preadj_bline-mean(preadj_bline),
           s_gpt_logp_adj=scale(gpt_logp_adj))]

formula_adj_logp = as.formula(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_logp + 
                               (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_logp || Subject) + 
                               (c_preadj_bline + s_pld20 + s_logfreq + s_logp || Item))
vars = c("s_gpt_logp_adj")
ret = get_topodat(dat, formula_adj_logp, vars, cores=2)
fwrite(ret, "adj_logp/topo adj_logp.txt", sep="\t")


# ---- Adj - log(gpt2.p) effect - plot (Figure 7) ----
#Figure 7 left (GGA)
eeg = eeg[items, on=c("Item", "exp_ver"), `:=`(gpt_logp_adj=i.gpt_logp_adj)]
eeg[, gpt_logp_adj.f:=factor(ifelse(gpt_logp_adj<median(gpt_logp_adj, na.rm=T), "low", "high"))]

dat = eeg[!is.na(gpt_logp_adj.f) & electrode=="MiPa", .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(electrode, ms, gpt_logp_adj.f)]
dat[, uV_bline:=uV-preadj_bline]

fig7L = ggplot(dat[electrode=="MiPa" & ms > - 100 & ms < 890], aes(x=ms, y=uV_bline, color=gpt_logp_adj.f)) +
  geom_erp() + 
  scale_color_manual("log(p)", values=c("#d95f02", "#7570b3")) +
  coord_cartesian(ylim=c(5.1, 5.1-erp_scale))


#Figure 7 middle-left (posterior)
x = readRDS("adj_logp/posterior_adj_logp_0.15.rds")

fig7ML = ggplot(x, aes(x=b_c_gpt_logp_adj, y="logp") ) + 
  geom_posterior(scale=1.2) + 
  annotate(geom="label", hjust=0, x=-0.05, y=10, label="log(p)", size=2, label.size=0, label.padding = unit(0.1, "lines"))


#Figure 7 middle-right (BFs)
x = fread("adj_logp/BF_adj_logp_.txt")
fig7MR = plot_BF(x, "sd", "log(p)\nNormal prior SD", 0.3)


#Figure 7 right (topomap)
topodat = fread("adj_logp/topo adj_logp.txt")
fig7R = ggplot(topodat, aes(x=x*150, y=y*150, fill=t_s_gpt_logp_adj, z=t_s_gpt_logp_adj)) + 
  geom_topomap() + 
  xlab("log(p)   300-500 ms")


fig7 = fig7L + fig7ML + fig7MR + fig7R + 
  plot_layout(design="AAAAAAAAABBBBBBBBBBCCCCCCCCCDDDDDDDD") #4.5 / 5 / 4.5 / 4
ggsave(fig7, file="Figure 7 R1.png", width=18, height=5, unit="cm", dpi=DPI)



# ---- Adj - Noun support effect - analysis ----
dat.adj.n400[, `:=`(Subject=factor(Subject),
                    Item=factor(Item),
                    s_logfreq=scale(log(freq_adj+1)),
                    s_pld20=scale(pld20_adj),
                    s_ctx_n_words=scale(ctx_n_words))]

w2v = fread("word2vec-google_ngrams.txt", col.names = c("adj", "noun", "w2v_cos"))
tops = unique(dat.adj.n400[noun_cond=="top", .(Item, noun)])
setnames(tops, "noun", "top_noun")
dat.adj.n400 = merge(dat.adj.n400, tops, by="Item")
dat.adj.n400 = merge(dat.adj.n400, w2v, by.x=c("adj", "top_noun"), by.y=c("adj", "noun"), all.x=T)
dat.adj.n400[, N_support:=(1-w2v_cos)*constr_preadj]
dat.adj.n400[, c_N_support:=N_support - mean(N_support)]

warmup = 1500
iter = 28500
cores = 14
dir = "adj_w2v/"
prefix = "adj_w2v_"

priors.adj = c(prior(normal( 0, 3), class = Intercept),
               prior(normal( 0.5, 0.3), class = b, coef="c_preadj_bline"),
               prior(normal( 0, 1), class = b, coef="s_pld20"),
               prior(normal( 0, 1), class = b, coef="s_logfreq"),
               prior(normal( 0, 1), class = b, coef="s_ctx_n_words"),
               prior(normal( 0, ef), class = b, coef="c_N_support"),
               prior(normal(10, 3), class = sigma),
               prior(normal( 0, 2), class = sd))

bm0 = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + 
            (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support || Subject) + 
            (c_preadj_bline + s_pld20 + s_logfreq + c_N_support || Item),
          prior   = priors.adj[-6,],
          warmup  = warmup,
          iter    = iter,
          cores   = cores,
          chains  = cores,
          control = list(adapt_delta = 0.9),
          save_pars = save_pars(all = TRUE),
          data    = dat.adj.n400, 
          file    = NULL
)

cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, 0))
cat(paste(capture.output(summary(bm0)), collapse="\n"))
cat("\n\n")

marg = bridge_sampler(bm0, silent = TRUE, cores=8)
saveRDS(marg, paste0(dir, "marg_", prefix, "null.rds"))
rm (bm0, marg)
gc()

for (ef in c(2, 4, 6, 8, 1, 10, 15, 20)) {
  stanvars = stanvar(ef, name="ef")
  bm = brm(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_N_support || Item)
           ,
           prior   = priors.adj,
           warmup  = warmup,
           iter    = iter,
           cores   = cores,
           chains  = cores,
           control = list(adapt_delta = 0.9),
           save_pars = save_pars(all = TRUE),
           stanvars = stanvars,
           data    = dat.adj.n400, 
           file    = NULL
  )
  
  if (ef == 6) {
    post = data.table(posterior_samples(bm, pars=c("b_c_N_support")))
    saveRDS(post, paste0(dir, "posterior_", prefix, ef, ".rds"))
    rm (post)
  }
  
  cat (sprintf("\n\n---\n%s: %.1f---\n", prefix, ef))
  cat(paste(capture.output(summary(bm)), collapse="\n"))
  cat("\n\n")
  
  marg = bridge_sampler(bm, silent = TRUE, cores=8)
  saveRDS(marg, paste0(dir, "marg_", prefix, ef, ".rds"))
  rm (bm, marg)
  gc()
}

ret = data.table()
marg0 = readRDS(paste0(dir, "marg_", prefix, "null.rds"))
for (ef in c(1, 2, 4, 6, 8, 10, 15, 20)) {
  filename = paste0(dir, "marg_", prefix,  ef, ".rds")
  if (!file.exists(filename)) next
  marg = readRDS(filename)
  BF = bayes_factor(marg, marg0)[[1]]
  ret = rbind(ret, data.table(sd=ef, BF=BF))
}
fwrite(ret, paste0(dir, "BF_", prefix, ".txt"), sep="\t")



#test if N_support improves fit over w2v_cos
dat.adj.n400[, s_w2v_cos:=scale(w2v_cos)]
dat.adj.n400[, s_constr_preadj:=scale(constr_preadj)]
m12 = lmer(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_w2v_cos + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_w2v_cos || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_N_support + s_w2v_cos || Item),
           data=dat.adj.n400, REML=F)
m1 = lmer(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_w2v_cos + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_w2v_cos || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_N_support + s_w2v_cos || Item),
           data=dat.adj.n400, REML=F)
anova(m1, m12)

#test if N_support improves fit over constr_preadj
m12 = lmer(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_constr_preadj + 
             (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_constr_preadj || Subject) + 
             (c_preadj_bline + s_pld20 + s_logfreq + c_N_support || Item),
           data=dat.adj.n400, REML=F)
m1 = lmer(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_constr_preadj + 
            (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + c_N_support + s_constr_preadj || Subject) + 
            (c_preadj_bline + s_pld20 + s_logfreq + c_N_support || Item),
          data=dat.adj.n400, REML=F)
anova(m1, m12)


#Prepare data for the scalp map
dat = eeg[!electrode %in% c("LE", "lhe", "rhe", "A2", "HEOG", "VEOG") & ms > 300 & ms <= 500, 
          .(uV=mean(uV), preadj_bline=mean(preadj_bline)), 
          by=.(Item, Subject, exp_ver, electrode)]
dat = merge(dat, items, by=c("Item", "exp_ver"))

dat = merge(dat, tops, by="Item")
dat = merge(dat, w2v, by.x=c("adj", "top_noun"), by.y=c("adj", "noun"), all.x=T)
dat[, N_support:=(1-w2v_cos)*constr_preadj]

dat[, `:=`(Subject=factor(Subject),
           Item=factor(Item),
           s_logfreq=scale(log(freq_adj+1)),
           s_pld20=scale(pld20_adj),
           s_ctx_n_words=scale(ctx_n_words),
           c_preadj_bline=preadj_bline-mean(preadj_bline),
           s_N_support=scale(N_support))]

formula_adj_N_support = as.formula(uV ~ c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_N_support + 
                                (c_preadj_bline + s_ctx_n_words + s_pld20 + s_logfreq + s_N_support || Subject) + 
                                (c_preadj_bline + s_pld20 + s_logfreq + s_N_support || Item))
vars = c("s_N_support")
ret = get_topodat(dat, formula_adj_N_support, vars, cores=2)
fwrite(ret, "adj_w2v/topo adj_w2v.txt", sep="\t")


# ---- Adj - Noun support effect - plot (Figure 8) ----
#Figure 8 left (GGA)
w2v = fread("word2vec-google_ngrams.txt", col.names = c("adj", "noun", "w2v_cos"))
tops = unique(items[noun_cond=="top", .(Item, noun)])
setnames(tops, "noun", "top_noun")
items = merge(items, tops, by="Item")
items = merge(items, w2v, by.x=c("adj", "top_noun"), by.y=c("adj", "noun"), all.x=T)
items[, N_support:=(1-w2v_cos)*constr_preadj]
items[, s_N_support:=scale(N_support)]
eeg = eeg[items, on=c("Item", "exp_ver"), `:=`(N_support=i.N_support)]
eeg[, w2v.f:=factor(ifelse(N_support<median(N_support, na.rm=T), "low", "high"))]

dat = eeg[!is.na(w2v.f) & electrode=="MiPa", .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(electrode, ms, w2v.f)] #GGA
dat[, uV_bline:=uV-preadj_bline]
fig8L = ggplot(dat[ms > -100 & ms < 890], aes(x=ms, y=uV_bline, color=w2v.f)) + 
  geom_erp() + 
  scale_color_manual("N support", values=c("#d95f02", "#7570b3")) +
  coord_cartesian(ylim=c(5.1, 5.1-erp_scale))


#Figure 8 middle-left (posterior)
x = readRDS("adj_w2v/posterior_adj_w2v_6.rds")
fig8ML = ggplot(x, aes(x=b_c_N_support, "")) + 
  geom_posterior(scale=1.2) + 
  annotate(geom="label", x=10, y=1.25, label="Noun support\nindex", size=2, label.size=0, label.padding = unit(0.1, "lines"))


#Figure 8 middle-right (BFs)
x = fread("adj_w2v/BF_adj_w2v_.txt")
fig8MR = plot_BF(x, "sd", "Noun support index\nNormal prior SD", 11)


#Figure 8 right (scalp map)
topodat = fread("adj_w2v/topo adj_w2v.txt")
fig8R = ggplot(topodat, aes(x=x*150, y=y*150, fill=t_s_N_support, z=t_s_N_support)) + 
  geom_topomap() + 
  xlab("N support   300-500 ms")


fig8L + fig8ML + fig8MR + fig8R + 
  plot_layout(design="AAAAAAAAABBBBBBBBBBCCCCCCCCCDDDDDDDD") #4.5 / 5 / 4.5 / 4
ggsave("Figure 8 R1.png", width=18, height=5, unit="cm", dpi=DPI)



# ------------- MASS UNIVARIATE TESTS - CLUSTER BASED ANALYSIS -----------------
# Code for cluster-based permutation test
# Written by Jakub Szewczyk (2018)

# Modeled on the algorithm presented in Maris and Oostenveld (2007)
# (the original Fieldtrip code uses a different clustering
# algorithm taken from SPM). Here I implemented my own
# clustering function that aggregates into clusters all 25-ms bins
# that are adjacent in time and space (cluster2 function).


#define electrodes' neighborhood
el_neighbors = list()
el_neighbors[["MiPf"]] = c("LLPf", "LMPf", "RMPf", "RLPf")
el_neighbors[["LLPf"]] = c("MiPf", "LMPf", "LDFr", "LLFr")
el_neighbors[["RLPf"]] = c("MiPf", "RMPf", "RDFr", "RLFr")
el_neighbors[["LMPf"]] = c("MiPf", "LLPf", "LDFr", "LMFr", "RMPf")
el_neighbors[["RMPf"]] = c("MiPf", "RLPf", "RDFr", "RMFr", "LMPf")
el_neighbors[["LDFr"]] = c("LMPf", "LLPf", "LLFr", "LDCe", "LMFr")
el_neighbors[["RDFr"]] = c("RMPf", "RLPf", "RLFr", "RDCe", "RMFr")
el_neighbors[["LLFr"]] = c("LLPf", "LLTe", "LDCe", "LDFr")
el_neighbors[["RLFr"]] = c("RLPf", "RLTe", "RDCe", "RDFr")
el_neighbors[["LMFr"]] = c("LMPf", "LDFr", "LMCe", "MiCe", "RMFr")
el_neighbors[["RMFr"]] = c("RMPf", "RDFr", "RMCe", "MiCe", "LMFr")
el_neighbors[["LDCe"]] = c("LDFr", "LLFr", "LLTe", "LDPa", "LMCe")
el_neighbors[["RDCe"]] = c("RDFr", "RLFr", "RLTe", "RDPa", "RMCe")
el_neighbors[["MiCe"]] = c("LMFr", "LMCe", "MiPa", "RMCe", "RMFr")
el_neighbors[["LMCe"]] = c("LMFr", "LDCe", "LDPa", "MiPa", "MiCe")
el_neighbors[["RMCe"]] = c("RMFr", "RDCe", "RDPa", "MiPa", "MiCe")
el_neighbors[["LLTe"]] = c("LDCe", "LLFr", "LLOc", "LDPa")
el_neighbors[["RLTe"]] = c("RDCe", "RLFr", "RLOc", "RDPa")
el_neighbors[["MiPa"]] = c("MiCe", "LMCe", "LMOc", "RMOc", "RMCe")
el_neighbors[["LDPa"]] = c("LMCe", "LDCe", "LLTe", "LLOc", "LMOc")
el_neighbors[["RDPa"]] = c("RMCe", "RDCe", "RLTe", "RLOc", "RMOc")
el_neighbors[["LMOc"]] = c("MiPa", "LDPa", "LLOc", "MiOc", "RMOc")
el_neighbors[["RMOc"]] = c("MiPa", "RDPa", "RLOc", "MiOc", "LMOc")
el_neighbors[["LLOc"]] = c("LDPa", "LLTe", "MiOc", "LMOc")
el_neighbors[["RLOc"]] = c("RDPa", "RLTe", "MiOc", "RMOc")
el_neighbors[["MiOc"]] = c("LMOc", "LLOc", "RLOc", "RMOc")

#1st level tests using LMER and baseline as a predictor
fun_1lvl_test_lmer = function(dat) {
  lm = lmer(data=dat, uV ~ c_preadj_bline + var + (1|Subject) + (1|Item))
  return (summary(lm)$coefficients["var", "t value"])
}

# function borrowed from the eyetrackingR package (https://github.com/jwdink/eyetrackingR)
# finds and labels runs of adjacent positive values
.label_consecutive <- function(vec) {
  vec = c(0,vec)
  vec[is.na(vec)] = 0
  out = c(cumsum(diff(vec)==1))
  vec = vec[-1]
  out[!vec] = NA
  out
}

# This function performs clustering in the time and channel domain
cluster2 = function (d, t_threshold) {
  ret = data.table()
  
  # we look separately for positive and negative clusters
  for (sign in c("pos", "neg")) {
    # level 1 analyses with abs(t) > t_threshold taken as significant
    if (sign=="pos") {
      d[, onoff:=V1 > t_threshold]
    } else {
      d[, onoff:=V1 < -t_threshold]
    }
    
    # Separately for each electrode, we look for clusters of significant
    # datapoints (so far time-domain only)
    d[, label:=.label_consecutive(onoff), by=.(electrode)]
    
    d2 = d[!is.na(label)] #remove non-significant datapoints 
    if (!nrow(d2)) next
    # Name each cluster such that it does not overlap with names of clusters
    # from other channels
    d2[, label:=paste(electrode, ".", label, sep="")] 
    setkeyv(d2, c("electrode", "bin"))
    
    current_els = unique(d2$electrode)
    
    # Perform spatial clustering based on the neighborhood matrix
    # defined in el_neighbors
    # (this presumably could be better optimized)
    for (el_from in names(el_neighbors)) {
      if (!el_from %in% current_els) next
      els_to_cluster = el_neighbors[[el_from]]
      for (el_to in els_to_cluster) {
        if (!el_to %in% current_els) next
        # Select a pair of neighboring electrodes
        d3 = dcast(d2[electrode %in% c(el_from, el_to)], bin ~ electrode, value.var="label") 
        setnames(d3, c("bin", "el_from", "el_to"))
        # Limit to clusters overlapping in time
        to_blend = unique(d3[!is.na(el_from) & !is.na(el_to), .(el_from, el_to)])
        for (rx in 1:nrow(to_blend)) {
          cl_from = to_blend[rx,1]
          cl_to = to_blend[rx,2]
          # Propagate/unify labels to clusters overlapping in time and
          # adjacent in space
          d2[label==cl_to, label:=cl_from]
        }
      }
    }
    d2[, direction:=sign]
    ret = rbind(ret, d2)
  }
  if (nrow(ret) > 0) setnames(ret, "V1", "mass")
  ret
}

# `eeg` is a data.table with EEG data in the long format.
# we look for clusters in the 100-700 ms time-window

# 24ms bins for faster computation and less noise and requiring certain level of temporal overlap among significant clusters
eeg[, ms2:=round(ms/24)*24]
dat = eeg[ms > 100 & ms < 700, .(uV=mean(uV), preadj_bline=mean(preadj_bline)), by=.(Subject, exp_ver, Item, electrode, ms2)]

dat = merge(dat, items, by=c("Item", "exp_ver"), all.x=T)
dat[, c_preadj_bline:=(preadj_bline-mean(preadj_bline))]
dat[, c_DKL:=scale(DKL, scale=F)]

setnames(dat, "ms2", "ms")
dat[, bin:=(ms/24)] #convert to bin number -- we need consecutive time values == bin numbers
gc()

setkeyv(dat, c("Subject", "electrode", "bin"))


# Identify clusters in the original data (unchanged condition labels)
t_threshold = 2
dat[, var:=c_DKL]

#run 1st level tests
dat_stat = dat[!is.na(var), fun_1lvl_test_lmer(.SD), by=.(bin, electrode)] 


org = cluster2(dat_stat, t_threshold) # perform clustering in time and channel space

#generate the report for all clusters found in the original data --> org2
org2a = dcast(org, direction + label ~ ., c(length, min, max, function(x) sum(abs(x))), value.var=c("bin", "mass"))[, .(direction, label, bin_min, bin_max, mass_function)]
setnames(org2a, "mass_function", "sum_statistic")
org2b = dcast(org, direction + label ~ ., function (x) paste(unique(x), collapse=","), value.var="electrode")
setnames(org2b, ".", "electrodes")
org2 = merge(org2a, org2b, by=c("label", "direction"), all.x=T)
org2[, ms_min:=bin_min*25]
org2[, ms_max:=bin_max*25]
org2[, .(label, ms_min, ms_max, sum_statistic, electrodes)] ## cluster report (including non-significant clusters)

# Generate the null distribution
n = 1000
max_mass_pos = rep(0, times=n)
max_mass_neg = rep(0, times=n)
subjects = unique(dat$Subject)

cl <- makeCluster(14, outfile="") #Adjust to your number of cores
registerDoParallel(cl)

rm (eeg); gc()
max_mass = foreach (x=1:n, .packages=c("lme4", "data.table"), .noexport=c("eeg", "dat_stat")) %dopar% {  # use 12 cores
  gc()
  set.seed(1214 + x)
  # Randomly select subjects for whom we reverse condition labels
  shuffler = data.table(Subject=subjects, rev=sample(0:1, size=length(subjects), replace=T)) 
  d2 = merge(dat, shuffler, by="Subject")
  # reverse the labels by inverting the sign of the deviation coded categorical predictor (-0.5, 0.5)
  d2[, var:=ifelse(rev==0, var, var * -1)] 
  dat_stat = d2[, fun_1lvl_test_lmer(.SD), by=.(bin, electrode)] # run t-tests on new labels
  rm (d2); gc()
  ret = cluster2(dat_stat, t_threshold) # perform clustering in time and channel space
  rm (dat_stat); gc()
  if (!nrow(ret)) return (list(0.0, 0.0))
  ret = ret[, sum(abs(mass)), by=.(label, direction)][, max(V1), by=.(direction)] # compute sum of test statistics for each cluster, select the maximal ones
  
  statsum = ret[direction=="pos", V1]
  if (length(statsum)) {
    max_mass_pos = statsum
  } else {
    max_mass_pos = 0
  }
  statsum = ret[direction=="neg", V1]
  if (length(statsum)) {
    max_mass_neg = statsum
  } else {
    max_mass_neg = 0
  }
  
  cat(sprintf("%d, max_mass_pos=%.0f, max_mass_neg=%.0f\n", x, max_mass_pos, max_mass_neg))
  return (list(max_mass_pos, max_mass_neg))
}
stopCluster(cl); rm(cl); gc()
max_mass = rbindlist(max_mass)
setnames(max_mass, c("pos", "neg"))
threshold_pos = max_mass[order(pos)][round(0.95 * n), pos]
threshold_neg = max_mass[order(neg)][round(0.95 * n), neg]

#No cluster is even close to the significance threshold
