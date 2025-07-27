library(shiny)
library(shinydashboard)        
library(nortest)
library(mvnormtest)
library(MASS)
library(shinyLP)
library(class)
library(gmodels)
library(caret)
library(rattle)
library(ranger)
library(klaR)
library(kernlab)

ui <- fluidPage(
        navbarPage(title = "Outpatient Appointments No-show Study",
                   tabPanel("Home",
                            jumbotron("Welcome", 
                                      paste("This is a web application for the patient's appointments No-show study.
                               For ease of use by staffs who are not familiar with R language"), 
                                      buttonLabel = "A CSCI-717 Project" )
                   ),
                   tabPanel("Data Sets", 
                            sidebarLayout(
                                    sidebarPanel(
                                            fileInput("file1", "Choose CSV File", accept=c('text/csv', 'text/comma-separated-values', 'text/plain', '.csv')),
                                            radioButtons("indata", "Choice:", choices = c("Full", "Columns")),
                                            selectInput("cols", "Choose the variable", choices = "", selected = " ", multiple = TRUE), 
                                            downloadButton('downloaddatset', "Download"),
                                            hr(),
                                            radioButtons("trans1", "Transformation:", choices = c("Not-Required", "log", "inverselog", "exponential", "lognormal", "standardize")),
                                            hr()
                                    ), 
                                    mainPanel(tableOutput("tab1"))
                            )
                   ), 
                   navbarMenu("Statistical Analysis",
                              tabPanel("Summary Statistics",
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("cols1", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                                       radioButtons("ssoption", "Select Option", choices = c("Summary", "Length", "Dim", "Type of", "Class"))
                                               ), 
                                               mainPanel(
                                                       fluidRow(
                                                               h3("Summary Statistics"),
                                                               div(
                                                                       verbatimTextOutput("summar")
                                                               )
                                                       )
                                               )
                                       )
                              ), 
                              tabPanel("Frequency Tables",
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("cols2", "Choose Varibale 1:", choices = "", selected = " ", multiple = TRUE),
                                                       selectInput("cols3", "Choose Varibale 2:", choices = "", selected = " ", multiple = TRUE)
                                               ), 
                                               mainPanel(
                                                       fluidRow(
                                                               h3("Frequency Tables"),
                                                               div(
                                                                       verbatimTextOutput("freq_tables")
                                                               )
                                                       )
                                               )
                                       )
                              ), 
                              
                              tabPanel("Plots",
                                       sidebarLayout(
                                               sidebarPanel(
                                                       radioButtons("plotoption", "Choose the Option:", choices = c("Histogram", "BarPlot", "Scatter", "Pie" )),
                                                       selectInput("cols6", "Choose Varibale 1:", choices = "", selected = " ", multiple = TRUE),
                                                       textInput("xaxisname", "Write X Axis Name"),
                                                       textInput("yaxisname", "Write Y Axis Name"),
                                                       textInput("title", "Write Title For the Graph")
                                               ), 
                                               mainPanel(
                                                       h3("Plots"),
                                                       fluidRow(
                                                               plotOutput("plot")
                                                       )
                                               )
                                       )
                                       
                              ),
                              
                              tabPanel("Statistical Tests", 
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("cols7", "Choose Varibale 1:", choices = "", selected = " ", multiple = TRUE),
                                                       selectInput("cols8", "Choose Varibale 2:", choices = "", selected = " ", multiple = TRUE),
                                                       radioButtons("normaltest", "Select Method:", choices = c("A-D-Test", "Shapiro", "KS-Test", "MV-Shapiro")),
                                                       hr(),
                                                       helpText("For more details visit:"),
                                                       a(href="https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test", "Anderson–Darling test"), br(),
                                                       a(href="https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test", "Shapiro–Wilk test"), br(),
                                                       a(href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test", "Kolmogorov–Smirnov test"), br(),
                                                       hr()
                                               ), 
                                               mainPanel(
                                                       h3("Statistical Tests"),
                                                       fluidRow(
                                                               div(
                                                                       plotOutput("qqp")
                                                               ),
                                                               div(
                                                                       verbatimTextOutput("normtest")
                                                               )
                                                       )
                                               )
                                       )
                              ),
                              
                              tabPanel("Correlation", 
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("cols9", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                                       selectInput("cols10", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                                       radioButtons("cormethod", "Select Method:", choices = c("Covariance", "KarlPearson", "Spearman", "Kendals")),
                                                       hr(),
                                                       helpText("For Details Visit:"),
                                                       a(href="https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient", "Karl Pearson Correlation Test"),
                                                       hr()
                                               ), 
                                               mainPanel(
                                                       h3("Covariance & Correlation"),
                                                       verbatimTextOutput("cor_t")
                                               )
                                               
                                       )
                                       
                              ),
                              
                              tabPanel("Regression & ANOVA", 
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("cols11", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                                       selectInput("cols12", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                                       radioButtons("regmethod", "Select Method:", choices = c("Fit", "Summary", "ANOVA")), 
                                                       hr(),
                                                       helpText("For more information please visit"),
                                                       a(href="https://en.wikipedia.org/wiki/Simple_linear_regression", "Simple Linear Regression"),
                                                       hr()
                                               ), 
                                               mainPanel(
                                                       h3("Regression & ANOVA"),
                                                       fluidRow(
                                                               div(
                                                                       verbatimTextOutput("regout")
                                                               ),
                                                               div(
                                                                       plotOutput("regplot")
                                                               )
                                                       )
                                               )
                                               
                                       )
                              )
                   ),
                   
                   navbarMenu("Predicition Models",
                              tabPanel("Logistic Regression",
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("logrvar", "Select Variable", choices = "", selected = ""),
                                                       textInput("logrprop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                                       # textInput("logryname", "Class Variable", value = "num", placeholder = "Class Variable"),
                                                       radioButtons("logroption", "Select Method", choices = c("Show Prop.", "Fit", "Coef.", "Pred. Accuracy")),
                                                       hr(),
                                                       helpText("Variable selected must be categorical."),
                                                       hr(),
                                                       a(href="https://en.wikipedia.org/wiki/Logistic_regression", "Logistic Regression")
                                               ),
                                               mainPanel(
                                                       div(verbatimTextOutput("logroutput"))
                                               )
                                       )
                              ),
                              
                              tabPanel("Decision Trees",
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("dtvar", "Response Variable", choices = "", selected = ""),
                                                       selectInput("dtvar2", "Explanatory Variable", choices = "", selected = "", multiple = TRUE),
                                                       textInput("dtprop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                                       textInput("dtyname", "Class Variable", value = "Char", placeholder = "Class Variable"),
                                                       radioButtons("dtoption", "Select Method", choices = c("No Option", "Table", "Show Prop.", "Train & Test Data", "Fit", "Predicted", "Pred. Accuracy")), 
                                                       radioButtons("dtplot", "Select Plot", choices = c("No Plot", "QPlot", "DTree")),
                                                       hr(),
                                                       helpText("Variable selected must be categorical and numerical."),
                                                       hr(),
                                                       a(href="https://en.wikipedia.org/wiki/Decision_tree", "Decision Trees")
                                               ),
                                               mainPanel(
                                                       div(verbatimTextOutput("dtoutput")),
                                                       div(plotOutput("dtplot"))     
                                               )
                                       )
                              ),
                              tabPanel("Random Forests", 
                                       sidebarLayout(
                                               sidebarPanel(
                                                       selectInput("rfvar", "Select Variable", choices = "", selected = ""),
                                                       
                                                       textInput("rfprop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                                       textInput("rfyname", "Class Variable", value = "old", placeholder = "Class Variable"),
                                                       radioButtons("rfoption", "Select Method", choices = c("No Option", "Table", "Show Prop.", "Train & Test Data", "Fit", "Summary", "Predicted", "Pred. Accuracy")),
                                                       hr(),
                                                       helpText("Variable selected must be categorical and numerical."),
                                                       hr(),
                                                       a(href="https://en.wikipedia.org/wiki/Random_forest", "Random Forest")
                                               ),
                                               mainPanel(
                                                       div(verbatimTextOutput("rfoutput"))
                                                       
                                               )
                                       )
                              )
                   ),
                   
                   tabPanel("About", 
                            sidebarLayout(
                                    sidebarPanel(
                                            "Project Information"
                                    ), 
                                    mainPanel(htmlOutput("text1"))
                            )
                   )
        )
)



server <- function(input, output, session) {
        
        # for DATASET TAB
        
        data_input <- reactive({
                infile <- input$file1
                req(infile)
                data.frame(read.csv(infile$datapath)) 
        })
        
        observeEvent(input$file1,{
                updateSelectInput(session, inputId = "cols", choices = names(data_input()))
        }
        )
        
        logno <- reactive({
                df <- data_input()
                x <- matrix(NA, length(df[, input$cols]), length(df[, input$cols][[1]]))
                for(i in 1:length(df[, input$cols])){
                        for(j in 1:length(df[, input$cols][[1]])){
                                x[i, j] <- dlnorm(df[, input$cols][[i]][j]) 
                        }
                }
                return(t(x))
        })
        
        standout <- reactive({
                df <- data_input()
                
                x <- matrix(NA, length(df[, input$cols]), length(df[, input$cols][[1]]))
                
                if(!is.list(df[, input$cols])){
                        df[, input$cols] <- list(df[, input$cols])
                }
                
                for(i in 1:length(df[, input$cols])){
                        
                        for(j in 1:length(df[, input$cols][[1]])){
                                x[i, j] <- df[, input$cols][[i]][j]-mean(df[, input$cols][[i]])/sd(df[, input$cols][[i]])
                        }
                }
                return(t(x))
                
        })
        
        logdata <- reactive({
                df <- data_input()
                ld <- log(df[, input$cols])
                return(ld)
        })
        
        invlogdata <- reactive({
                df <- data_input()
                ild <- 1/log(df[, input$cols])
                return(ild)
        })
        
        expdata <- reactive({
                df <- data_input()
                expd <- log(df[input$cols])
                return(expd)
        })
        
        
        output$tab1 <- renderTable(
                {
                        df <- data_input()
                        
                        if (input$indata == "Full"){
                                print(df)
                        } else if(input$trans1 == "Not-Required"){
                                data <- df[, input$cols]
                                print(data)
                        } else if(input$trans1 == "log"){
                                logdata()
                                
                        } else if(input$trans1 == "inverselog"){
                                invlogdata()
                        } else if(input$trans1 == "exponential"){
                                expdata()
                        } else if(input$trans1 == "lognormal"){
                                logno()
                        } else if(input$trans1 == "standardize"){
                                standout()
                        }
                        
                }
        )
        
        output$downloaddatset <- downloadHandler(
                
                filename <- function(){
                        paste("data-", Sys.Date(), ".csv", sep = "")
                },
                
                content <- function(file){
                        df <- data_input()
                        if(input$trans1 == "log"){
                                write.csv(logdata(), file, row.names = TRUE)
                        } else if(input$trans1 == "inverselog"){
                                write.csv(invlogdata(), file, row.names = TRUE)
                        } else if(input$trans1 == "exponential"){
                                write.csv(expdata(), file, row.names = TRUE)
                        } else if(input$trans1 == "lognormal"){
                                write.csv(logno(), file, row.names = TRUE)
                        } else if(input$trans1 == "standardize"){
                                write.csv(standout(), file, row.names = TRUE)
                        }
                        
                }
                
        )
        
        # summary statistics
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols1", choices = names(data_input()))
        }
        )
        
        summ <- reactive({
                var1 <- data_input()[,input$cols1]
                
                if (input$ssoption == "Summary"){
                        su <- summary(var1)
                        return(su)
                } else if (input$ssoption == "Length"){
                        return(length(var1))
                } else if(input$ssoption == "Dim"){
                        return(dim(var1))
                } else if (input$ssoption == "Type of"){
                        return(typeof(var1))
                } else if(input$ssoption == "Class"){
                        return(class(var1))
                }
        })
        
        output$summar <- renderPrint({
                
                if (input$ssoption == "Summary"){
                        summ()
                } else if (input$ssoption == "Length"){
                        summ()
                } else if(input$ssoption == "Dim"){
                        summ()
                } else if (input$ssoption == "Type of"){
                        summ()
                } else if(input$ssoption == "Class"){
                        summ()
                }
        })
        
        # frequency tab
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols2", choices = names(data_input()))
                updateSelectInput(session, inputId = "cols3", choices = names(data_input()))
        }
        )
        
        freq <- reactive({
                var1 <- data_input()[,input$cols2]
                var2 <- data_input()[,input$cols3]
                fre <- table(var1, var2)
                return(fre)
        })
        
        output$freq_tables <- renderPrint({
                freq()
        })
        
        # Cross tabulation
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols4", choices = names(data_input()))
                updateSelectInput(session, inputId = "cols5", choices = names(data_input()))
        }
        )
        
        cross <- reactive({
                var1 <- data_input()[,input$cols4]
                var2 <- data_input()[,input$cols5]
                
                cro <- chisq.test(var1, var2)
                return(cro)
        })
        
        output$chi_t <- renderPrint({
                cross()
                
        })
        
        # Plots 
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols6", choices = names(data_input()))
        }
        )
        
        output$plot <- renderPlot({
                df <- data_input()
                if(input$plotoption == "Histogram"){
                        hist(df[, input$cols6], freq = FALSE, xlab = input$xaxisname, ylab = input$yaxisname, main = input$title); lines(density(df[, input$cols6]), col = "red", lwd = 1.5)
                } else if(input$plotoption == "BarPlot"){
                        barplot(df[, input$cols6], xlab = input$xaxisname, ylab = input$yaxisname, main = input$title)
                } else if(input$plotoption == "Scatter"){
                        scatter.smooth(df[, input$cols6], xlab = input$xaxisname, ylab = input$yaxisname, main = input$title)
                } else {
                        pie(table(df[, input$cols6]))
                }
        })
        
        # Statistical Tests
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols7", choices = names(data_input()))
                updateSelectInput(session, inputId = "cols8", choices = names(data_input()))
        }
        )
        
        output$qqp <- renderPlot({
                df <- data_input()
                qqnorm(df[, input$cols7]);qqline(df[, input$cols7])
        })
        
        adt <- reactive({
                df <- data_input()
                var <- df[, input$cols7]
                ad <- ad.test(var)
                return(ad)
        })
        
        sht <- reactive({
                df <- data_input()
                var <- df[, input$cols7]
                sh <- shapiro.test(var)
                return(sh)
        })
        
        kst <- reactive({
                df <- data_input()
                var1 <- df[, input$cols7]
                var2 <- df[, input$cols8]
                ks <- ks.test(var1, var2)
                return(ks)
        })
        
        mvst <- reactive({
                df <- data_input()
                var1 <- df[, input$cols7]
                var2 <- df[, input$cols8]
                return(mshapiro.test(t(as.data.frame(var1, var2))))
        })
        
        output$normtest <- renderPrint({
                
                if(input$normaltest == "A-D-Test"){
                        print(adt())
                } else if(input$normaltest == "Shapiro"){
                        print(sht())
                } else if(input$normaltest == "KS-Test"){
                        print(kst())
                } else if(input$normaltest == "MV-Shapiro"){
                        print(mvst())
                }
                
        }
        
        )
        # correlation & regression 
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "cols9", choices = names(data_input()))
                updateSelectInput(session, inputId = "cols10", choices = names(data_input()))
        }
        )
        
        cortest <- reactive({
                var1 <- data_input()[,input$cols9]
                var2 <- data_input()[,input$cols10]
                
                if (input$cormethod == "Covariance"){
                        return(cov(var1, var2))
                } else if (input$cormethod == "KarlPearson"){
                        return(cor.test(var1, var2, method = "pearson"))
                } else if(input$cormethod == "Spearman"){
                        return(cor.test(var1, var2, method="spearman"))
                } else {
                        return(cor.test(var1, var2, method="kendall"))
                }
        }
        )
        
        output$cor_t <- renderPrint({
                
                cortest()
        })
        
        
  # Logistic Regression
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "logrvar", choices = names(data_input()))
        }
        )
        
        logrout <- reactive({
                
                df <- data_input()
                
                var <- input$logrvar
                
                Train <- createDataPartition(df[, var], p=as.numeric(input$logrprop), list=FALSE)
                training <- df[Train, ]
                testing <- df[-Train, ]
                
                trainprop <- nrow(training)/(nrow(testing)+nrow(training))
                
                # var1 <- input$logryname
                
                mod_fit <- train(as.formula(paste(var, "~", ".")),  data=training, method="glm", family="binomial")
                
                expout <- exp(coef(mod_fit$finalModel))
                
                if (input$logroption == "Show Prop."){
                        return(trainprop)
                } else if (input$logroption == "Fit"){
                        return(mod_fit)
                }
                
                if (input$logroption == "Coef."){
                        return(data.frame(expout))
                }
                
                predout <- predict(mod_fit, newdata=testing)
                # predoutproba <- predict(mod_fit, newdata=testing, type="prob")
                accuracy <- table(predout, testing[, input$logrvar])
                out <- sum(diag(accuracy))/sum(accuracy)
                
                confmat <- confusionMatrix(round(predout), testing[, input$logrvar])
                
                if (input$logroption == "Pred. Accuracy"){
                        return(confmat)
                }
                return(var1)
        })
        
        
        output$logroutput <- renderPrint({
                
                if(input$logroption == "Coef."){
                        logrout()
                } else if (input$logroption == "Show Prop."){
                        logrout()
                } else if (input$logroption == "Fit"){
                        logrout()
                } else if (input$logroption == "Pred. Accuracy"){
                        logrout()
                }
                
        })
        
        
        # DECISION TREE
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "dtvar", choices = names(data_input()))
                updateSelectInput(session, inputId = "dtvar2", choices = names(data_input()))
        }
        )
        
        dtout <- reactive({
                
                df <- data_input()
                tab = table(df[, input$dtvar])
                
                if (input$dtoption == "Table"){
                        return(tab)
                }
                
                index = createDataPartition(y=df[, input$dtvar], p=0.7, list=FALSE)
                
                train.set = df[index,]
                test.set = df[-index,]
                
                if (input$dtoption == "Train & Test Data"){
                        return(list(head(train.set), head(test.set)))
                }
                
                if (input$dtoption == "Show Prop."){
                        return(dim(train.set))
                }
                
                var <- input$dtvar
                
                brest.tree = train(as.formula(paste(var, "~", ".")),
                                   data=train.set,
                                   method="rpart",
                                   trControl = trainControl(method = "cv"))
                
                if (input$dtplot == "QPlot"){
                        
                        plot(brest.tree$finalModel, uniform=TRUE, main="Classification Tree"); text(brest.tree$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
                }
                
                
                if (input$dtoption == "Fit"){
                        return(brest.tree)
                }
                
                
                if (input$dtplot == "DTree"){
                        fancyRpartPlot(brest.tree$finalModel)
                }
                
                pred <- predict(brest.tree, test.set)
                out <- confusionMatrix(pred, test.set[, "Class_num"])
                
                if (input$dtoption == "Predicted"){
                        return(data.frame(pred))
                }
                
                if (input$dtoption == "Pred. Accuracy"){
                        return(out)
                }
                
        })
        
        output$dtoutput <- renderPrint({
                dtout()
        })
        
        output$dtplot <- renderPlot({
                if (input$dtplot == "QPlot"){
                        dtout()
                } else if (input$dtplot == "DTree"){
                        dtout()
                } else if (input$dtoption == "Pred. Accuracy"){
                        dtout()
                } else if (input$dtoption == "Predicted"){
                        dtout()
                }
        })  
        
        # RANDOM FOREST
        
        observeEvent(input$file1, {
                updateSelectInput(session, inputId = "rfvar", choices = names(data_input()))
        }
        )
        
        rfout <- reactive({
                df <- data_input()
                
                if (input$rfoption == "Table"){
                        return(table(df[, input$rfvar]))
                }
                
                # train_index <- sample(1:nrow(df), as.numeric(input$rfprop) * nrow(df))
                
                prop <- as.numeric(input$rfprop)
                
                train_set <- df[1:(nrow(df)*prop),]
                test_set <- df[-(1:(nrow(df)*prop)),]
                
                if (input$rfoption == "Show Prop."){
                        return(dim(train_set)[1]/dim(df)[1])
                }
                if (input$rfoption == "Train & Test Data"){
                        return(list(head(train_set), head(test_set), dim(train_set), dim(test_set)))
                }
                var <- input$rfvar
                rf_fit <- randomForest::randomForest(as.formula(paste(var, "~", ".")), data = train_set, importance = TRUE, proximity = TRUE)
                
                if (input$rfoption == "Fit"){
                        return(rf_fit)
                }
                
                if (input$rfoption == "Summary"){
                        return(summary(rf_fit))
                }
                
                rf_pred <- predict(rf_fit, test_set)
                
                out <- confusionMatrix(round(rf_pred), as.numeric(test_set[, input$rfvar]))
                
                if (input$rfoption == "Predicted"){
                        return(data.frame(rf_pred))
                }
                if (input$rfoption == "Pred. Accuracy"){
                        return(out)
                }
                
                # return(out)
        })
        output$rfoutput <- renderPrint({
                rfout()
        })  
        
        # Project Information 
        
        output$text1 <- renderText({
                str1 <- paste("Developer: Fangzheng Yuan") 
                str2 <- paste("Email: fangzheng.yuan@ndus.edu") 
                str3 <- paste("Lecture: Dr. Pratap Kotala")
                str4 <- paste("Address:North Dakota State University, Fargo, ND 58102")
                HTML(paste(str1, str2, str3, str4,sep = '<br/>'))
        })
}

shinyApp(ui, server)
