#####################
####Data Preparation#
#####################

setwd('/Users/ryotarohiraki/Desktop/Spring2025/Economics of Education/final paper(35%)/data')

library(haven)       # read data
library(DoubleML)    # Double Machine Learning
library(mlr3)        # ML models
library(mlr3learners) 
library(data.table)  # data management
library(sandwich)    # robust sd
library(lmtest)      # test reg results
library(ranger)
library(tidyverse)
library(dplyr)

########################
###data cleaning process
########################

#read a personal dataset
raw_cps2024 <- read_csv("asecpub24csv/pppub24.csv",col_types = cols(PERIDNUM = col_character())) |> zap_formats()

#read a household dataset
raw_hcps2024 <- read_csv("asecpub24csv/hhpub24.csv", col_types = cols(H_IDNUM = col_character())) |> zap_formats()

#avoid ids to be a exponential form by importing as character

#GESTFIPS- state code (ONLY EXISTS IN HOUSEHOLD SURVEY)

#A_HGA- educational attainment
#A_AGE- age
#A_SEX- sex(1-male, 2-female)
#A_MARITL- martial status
#PRDTRACE- race
#A_HRSPAY- hourly wage
#PERIDNUM- 22 digits unique person identifier number.
#A_ERNLWT- earn weight (two implied decimals)
#A_UNMEM- labour union
#A_WKSTAT- full time status (==2 full time worker)

#H_IDNUM(20 digits)- Household number(in household data)
#PERIDNUM(22 digits) - Same number as household number(in personal data) + 2 digits
#They are to link household to a person to identify its state.


#take first 20 digits of PERIDNUM
raw_cps2024 <- raw_cps2024 %>%
  mutate(H_IDNUM = (substr(PERIDNUM, 1, 20)))

#connect two dataset by household number and select variables
cps_analysis <- raw_cps2024 %>%
  left_join(raw_hcps2024 %>% select(H_IDNUM, GESTFIPS), by = "H_IDNUM") %>%
  select(GESTFIPS, PERIDNUM, H_IDNUM, PEARNVAL, HRSWK, WKSWORK, A_WKSTAT, A_HGA, A_AGE, A_SEX, A_MARITL, PRDTRACE, PEHSPNON, A_HRSPAY, A_ERNLWT, A_UNMEM) %>%
  rename("state_id" = "GESTFIPS",
         "pp_id" = "PERIDNUM",
         "hh_id" = "H_IDNUM",
         "annual_income" = "PEARNVAL",
         "work_hpw" = "HRSWK",
         "work_wpy" = "WKSWORK",
         "education" = "A_HGA",
         "fulltime" = "A_WKSTAT",
         "age" = "A_AGE",
         "sex" = "A_SEX",
         "martial" = "A_MARITL",
         "race" = "PRDTRACE",
         "hispanic" = "PEHSPNON",
         "hwage" = "A_HRSPAY",
         "earn_wgt" = "A_ERNLWT",
         "lunion" = "A_UNMEM") %>%
  mutate(female = if_else(sex == 2, 1, 0),
         education = case_when(
           education == 31 ~ 0,
           education == 32 ~ 2,
           education == 33 ~ 6,
           education == 34 ~ 8,
           education == 35 ~ 9,
           education == 36 ~ 10,
           education == 37 ~ 11,
           education == 38 ~ 12,   # 12 but not graduated
           education == 39 ~ 12,
           education == 40 ~ 13,
           education %in% c(41, 42) ~ 14,
           education == 43 ~ 16,
           education == 44 ~ 18,
           education == 45 ~ 20,
           education == 46 ~ 21,
           TRUE ~ NA_real_
         ),
         experience = age - education - 6,
         education2 = education^2, #newly added variable
         college = if_else(education >= 16, 1, 0), #newly added
         experience2 = experience^2,
         earn_wgt = earn_wgt / 100,
         hispanic = if_else(hispanic == 1, 1, 0),
         afr_amer = if_else(race == 2, 1, 0),
         asian = if_else(race == 4, 1, 0),
         in_union = if_else(martial %in% c(1,2,3), 1, 0),
         lunion = if_else(lunion == 1, 1, 0),
         fulltime = if_else(fulltime == 2, 1, 0))


#hourly wage is A_HRSPAY if paied hourly, or annual income/ hour worked per year
cps_analysis <- cps_analysis %>%
  mutate(
    annual_hours = work_hpw * work_wpy,
    hwage = case_when(
      !is.na(hwage) & hwage > 0 ~ hwage / 100, #2 decimal places
      !is.na(annual_income) & !is.na(work_hpw) & !is.na(work_wpy) & work_hpw > 0 & work_wpy > 0 ~ annual_income / annual_hours,
      TRUE ~ NA_real_
    )
  )

#drop non fulltime workers (<35 hrs a week, <10 weeks a year),                                                                                                                                                                                                         drop missing values in hwage and education
cps_analysis <- cps_analysis %>%
  filter(fulltime == 1,
         earn_wgt > 0, #apply earn weight (drop the observation to 10,000)
         !is.na(hwage), 
         work_hpw >= 35, work_wpy >= 10, hwage < 100, hwage > 7.5,    #get rid of those whose hwage is <7.5 (Federal min wage) and > $100(outlier), worked less than 5 hrs per week/week per year
       !is.na(education), education > 0,
       experience > 0) %>% #get rid of negative experience           
  mutate(log_hwage = log(hwage),
         w = earn_wgt / mean(earn_wgt)) # set standardized earning weight

#combine state name with the id
#state_code <- read_csv("asecpub24csv/state_code.csv") |> zap_formats()

# cps_analysis <- cps_analysis %>%
#   left_join(state_code, by = "state_id")


######################
#visuals##############
######################

#Descriptive Statistics
descriptive <- cps_analysis %>%
  summarise(across(c(education, hwage, experience, female, experience, 
                     afr_amer, asian, hispanic, in_union, lunion),
                   list(mean = ~mean(., na.rm = TRUE),
                        min = ~min(., na.rm = TRUE),
                        max = ~max(., na.rm = TRUE),
                        obs = ~sum(!is.na(.)),
                        std_dev = ~sd(., na.rm = TRUE))
  ))

#scatter plot edu and log_hwage
cps_analysis %>%
  ggplot(aes(x = education, y = log_hwage)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Education", y = "Log hourly wage") +
  theme_minimal()

#logwage distribution
cps_analysis %>%
  ggplot(aes(x = log_hwage)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(x = "Log hourly wage") +
  theme_minimal()
