rm(list=ls())


library(tidyverse)
library(caret)
library(ranger)

#source("set-data-directory.R") 
source("ch00-tech-prep/theme_bg.R")
source("ch00-tech-prep/da_helper_functions.R")

data_in <- 'C:/Workspace/R_directory/CEU/Data_Analysis_3/da_data_repo/hotels-europe'
data_out <- paste0(data_in,'/out/')

create_output_if_doesnt_exist(data_out)



# Read in data - merge price and features ---------------------------------

df_price <- read_csv(paste0(data_in,'/clean/hotels-europe_price.csv'))
checker <- df_price %>%  group_by(year,month,weekend) %>%  summarise(count = n())
checker
df_price = df_price[df_price$year == 2018 & df_price$month == 6, ]
df_features <- read_csv(paste0(data_in,'/clean/hotels-europe_features.csv'))

df = merge(df_price,df_features)

# Look at missing values and duplicates

plot_Missing <- function(data_in, title = NULL){
    temp_df <- as.data.frame(ifelse(is.na(data_in), 0, 1))
    temp_df <- temp_df[,order(colSums(temp_df))]
    data_temp <- expand.grid(list(x = 1:nrow(temp_df), y = colnames(temp_df)))
    data_temp$m <- as.vector(as.matrix(temp_df))
    data_temp <- data.frame(x = unlist(data_temp$x), y = unlist(data_temp$y), m = unlist(data_temp$m))
    ggplot(data_temp) + geom_tile(aes(x=x, y=y, fill=factor(m))) + scale_fill_manual(values=c("white", "black"), name="Missing\n(0=Yes, 1=No)") + theme_light() + ylab("") + xlab("") + ggtitle(title)
}

plot_Missing(df[,colSums(is.na(df))>0])

colSums(is.na(df))
colSums(is.na(df))/nrow(df)

df[duplicated(df)]



# Feature engineering -----------------------------------------------------

head(df)

# Cleaning decision: 
#  - Impute:   rating(median), ratingreviewcount(mean) 
#  - Drop : stars, ratingta, ratingta_count, year, month, holiday
#  - Drop NA observations: accomodation type
#  - Drope observations where price is higher than 1500 - these are extreme values that affect our predictions by a large margin
# Groupped variables: 
#  - acomodation type - create 4 factors and drop the rest
#  - offers - if more than 15% we group them togethe


df <-
df %>% 
    mutate(
        accommodation_type = 
            ifelse(accommodation_type %in% c('Bed and breakfast', 'Hostel','Motel' ),'hostel',
                   ifelse(accommodation_type == 'Hotel','hotel',
                          ifelse(accommodation_type %in% c('Apart-hotel','Apartment', 'Vacation home Condo' ), 'apartment', 
                                 ifelse(accommodation_type %in% c('Guest House', 'Inn', 'Pension', 'Cabin / Lodge', 'Cottage', 'Country House'), 'guest_house', accommodation_type)))),
        offer_cat = 
            factor(ifelse(offer_cat %in% c('15-50% offer', '50%-75% offer', '75%+ offer' ),'15%+ offer',offer_cat)),
        offer_cat = gsub(" no offer$| offer$", "", offer_cat),
        offer = NULL,
        stars = NULL,
        ratingta = NULL,
        ratingta_count = NULL,
        year = NULL, 
        month = NULL,
        holiday = NULL,
        weekend = NULL,
        center1label = NULL,
        nnights = NULL,
        flag_rating = ifelse(is.na(rating), 1,0),
        rating = ifelse(is.na(rating),as.numeric(median(rating, na.rm=T)),as.numeric(rating)),
        flag_rating_reviewcount = ifelse(is.na(rating_reviewcount), 1,0),
        rating_reviewcount = ifelse(is.na(rating_reviewcount), mean(rating_reviewcount, na.rm=TRUE),rating_reviewcount ),
        price = as.numeric(price),
        distance = as.numeric(distance),
        rating_reviewcount = as.numeric(rating_reviewcount),
        distance_alter = as.numeric(distance_alter),
        scarce_room =factor(scarce_room)
    ) %>% 
    filter(
        !is.na(accommodation_type),
        accommodation_type %in% c('hotel','apartment', 'guest_house', 'hostel')
    ) %>% 
    mutate(accommodation_type = factor(accommodation_type))

df<- df %>% filter(price < 1500)


# Modeling ----------------------------------------------------------------
unique(df$scarce_room)
#Splitting holdout set 

set.seed(7)
train_indices <- as.integer(createDataPartition(df$price, p = 0.7, list = FALSE))
df_train <- df[train_indices, ]
df_holdout <- df[-train_indices, ]


# do 5-fold CV
train_control <- trainControl(method = "cv",
                              number = 5,
                              verboseIter = FALSE)

# set tuning
tune_grid <- expand.grid(
    .mtry = c(3,9, 11,13,15),
    .splitrule = "variance",
    .min.node.size = c( 5, 7, 10,15 )
)

vars_not_to_include = c('hotel_id', 'country', 'city_actual', 'center2label','neighbourhood','price' )
predictors = colnames(df)[!colnames(df) %in% vars_not_to_include]

predictors
# run random forrest
set.seed(7)
system.time({
    rf_model <- train(
        formula(paste0("price ~", paste0(predictors, collapse = " + "))),
        data = df_train,
        method = "ranger",
        trControl = train_control,
        tuneGrid = tune_grid,
        importance = "impurity"
    )
})

saveRDS(rf_model, 'rf_model.rds')

#check results
rf_tuning_model <- rf_model$results %>%
    dplyr::select(mtry, min.node.size, RMSE) %>%
    dplyr::rename(nodes = min.node.size) %>%
    spread(key = mtry, value = RMSE)

result <- matrix(
    c(rf_model$finalModel$mtry,rf_model$finalModel$min.node.size),
    nrow=1, ncol=2,
    dimnames = list(c("RF model"),
                c("Min vars","Min nodes"))
)
rf_tuning_model
result


cv_results_table <- rf_model$results %>% as.data.frame %>% select(c(mtry, min.node.size, RMSE))

#evaluate model on the holdouot set

data_holdout_w_prediction <- df_holdout %>%
    mutate(predicted_price = predict(rf_model, newdata = df_holdout))


d <- data_holdout_w_prediction %>%
    dplyr::summarise(
        rmse = RMSE(predicted_price, price),
        mean_price = mean(price),
        rmse_norm = RMSE(predicted_price, price) / mean(price)
    )


colnames(cv_results_table) <- c('Vars to select', 'Minimum node size', 'RMSE')

cv_results_table[cv_results_table$`Vars to select` == 13 & cv_results_table$`Minimum node size` == 15,]$RMSE

df$pred_price <-  predict(rf_model, newdata = df)

cv_rmse <- cv_results_table[cv_results_table$`Vars to select` == 13 & cv_results_table$`Minimum node size` == 15,]$RMSE
cv_mean_p <- mean(df_train$price) 
cv_line <- c('CV RMSE',cv_rmse, cv_mean_p, cv_rmse/cv_mean_p)

holdout_rmse <- RMSE(data_holdout_w_prediction$predicted_price,data_holdout_w_prediction$price)
holdout_mean_p <- mean(data_holdout_w_prediction$price)
holdout_line <- c('Holdout RMSE',holdout_rmse, holdout_mean_p, holdout_rmse/holdout_mean_p)

total_rmse <- RMSE(df$pred_price,df$price)
total_mean_p <- mean(df$price)
total_line <- c('All data RMSE',total_rmse, total_mean_p, total_rmse/total_mean_p)

results_sum <- rbind(cv_line,holdout_line,total_line) %>% as.data.frame

rownames(results_sum) = c()
colnames(results_sum) = c(' ', 'RMSE', 'Mean Price', 'RMSE - norm')


ggplot(data = data_holdout_w_prediction, aes(x = price)) +
    theme_bw() + 
    geom_point(aes(x = price, y = predicted_price), color='blue') +
    geom_abline(color = 'red') + xlim(0,1500) + ylim(0,1500)


# Variable importance plots 

var_imp <- importance(rf_model$finalModel)/1000
var_imp_df <-
    data.frame(varname = names(var_imp),imp = var_imp) %>%
    arrange(desc(imp)) %>%
    mutate(imp_percentage = imp/sum(imp))

var_imp_df

p1 <- ggplot(var_imp_df[1:10,], aes(x=reorder(varname, imp), y=imp_percentage)) +
    geom_point(colour="navyblue", size=1) +
    geom_segment(aes(x=varname,xend=varname,y=0,yend=imp_percentage), colour="navyblue", size=0.75) +
    ylab("Importance (Percent)") +
    xlab("Variable Name") +
    coord_flip() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_bw() +
    theme(axis.text.x = element_text(size=8), axis.text.y = element_text(size=8),
          axis.title.x = element_text(size=8), axis.title.y = element_text(size=8))



cities <- subset(var_imp_df$varname, grepl('city', var_imp_df$varname))

group_var_imp_df <- 
var_imp_df %>%  
    mutate(group = ifelse(varname %in% cities, 'city' ,varname)) %>%
    group_by(group) %>% 
    summarise(group_imp_sum = sum(imp_percentage))

    
p2<- ggplot(group_var_imp_df, aes(x=reorder(group, group_imp_sum), y=group_imp_sum)) +
    geom_point(colour="navyblue", size=1) +
    geom_segment(aes(x=group,xend=group,y=0,yend=group_imp_sum), colour="navyblue", size=0.75) +
    ylab("Importance (Percent)") +
    xlab("Variable Name") +
    coord_flip() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_bw() +
    theme(axis.text.x = element_text(size=8), axis.text.y = element_text(size=8),
          axis.title.x = element_text(size=8), axis.title.y = element_text(size=8))

nrow(var_imp_df)
