################################################################################
#' @title calcAbsRatio

#' @author
#' Robert Hensley \email{hensley@battelleecology.org} \cr

#' @description This function calculates the absorbance ratio for two user-
#' specified wavelengths from National Ecological Observatory Network (NEON)
#' water chemistry data.

#' @param absorbanceData User input of the table of NEON absorbance data [data.frame]
#' @param concentrationData User input of the table of NEON concentration data [data.frame]
#' @param wavelength1 User input of the wavelength to be used in the numerator [numeric]
#' @param wavelength2 User input of the wavelength to be used in the denominator [numeric]
#' @param correctFe User input of whether correction for overlapping absorbption of Fe(III)
#' should also be included. See README for more information. Defaults to FALSE. [logical]

#' @import tidyverse

#' @return This function returns a table of absorbance ratio values

#' @references
#' License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

#' @keywords NEON, DOC, SUVA, UV Absorbance

#' @examples
#' #Using an example file
#' #outputData <- calcAbsRatio(
#' #absorbanceData=swc_externalLabAbsorbanceScan,
#' #concentrationData=swc_externalLabDataByAnalyte,
#' #wavelength1 = 300
#' #wavelength2 = 400
#' #correctFe=TRUE)

#' @export

# changelog and author contributions / copyrights
################################################################################
calcAbsRatio<-function(
    absorbanceData,
    concentrationData=FALSE,
    wavelength1,
    wavelength2,
    correctFe=FALSE){

  if(nrow(absorbanceData) < 1){
    print("Error: No absorbance data")
    return(NULL)
  }

  if(wavelength1<220|wavelength1>600){
    print("Error: Wavelength 1 is outside 200-600nm range of absorbance scan")
    return(NULL)
  }
  if(wavelength2<220|wavelength2>600){
    print("Error: Wavelength 2 is outside 200-600nm range of absorbance scan")
    return(NULL)
  }

  # Isolates specified wavelengths if even, or surrounding wavelengths if odd.
  if(wavelength1 %% 2 == 0){
    absorbanceData1<-absorbanceData[(absorbanceData$wavelength==wavelength1),]
  }
  if(wavelength1 %% 2 != 0){
    absorbanceData1<-absorbanceData[(absorbanceData$wavelength==wavelength1-1|absorbanceData$wavelength==wavelength1+1),]
  }
  if(wavelength2 %% 2 == 0){
    absorbanceData2<-absorbanceData[(absorbanceData$wavelength==wavelength2),]
  }
  if(wavelength2 %% 2 != 0){
    absorbanceData2<-absorbanceData[(absorbanceData$wavelength==wavelength2-1|absorbanceData$wavelength==wavelength2+1),]
  }
  absorbanceData<-rbind(absorbanceData1,absorbanceData2)

  # Averages replicate absorbance scans
  absorbanceAveraged<-absorbanceData
  absorbanceAveraged$sampleID.wavelength<-paste(absorbanceAveraged$sampleID, absorbanceAveraged$wavelength, sep=".")
  absorbanceAveraged<-absorbanceAveraged |> dplyr::group_by(sampleID.wavelength) |> dplyr::summarize(
    sampleID=unique(sampleID),domainID=unique(domainID),siteID=unique(siteID),
    collectDate=unique(collectDate),wavelength=unique(wavelength),absorbance=mean(decadicAbsorbance))

  absorbanceAveraged$sampleID.wavelength<-NULL

  # No concentration required if Fe correction not performed
  combinedData<-absorbanceAveraged

  # Adds Fe concentrations if input condition is true
  if(correctFe == TRUE){
    Fe<-concentrationData[(concentrationData$analyte=="Fe"),]
    if(nrow(Fe) < 1){
      print("Error: No concentration data")
      return(NULL)
    }
    Fe<-Fe[,c("sampleID","analyteConcentration")]
    colnames(Fe)<-c("sampleID","Fe")
    combinedData<-merge(absorbanceAveraged, Fe,by.x="sampleID",by.y="sampleID")
    combinedData$absorbanceFe<-combinedData$Fe*((-0.00000044*(combinedData$wavelength^2))-(0.00007755*combinedData$wavelength)+0.11337671)
    combinedData$absorbanceCorrected<-combinedData$absorbance-combinedData$absorbanceFe
  }


  # Performs calculation without any corrections
    abs1<-combinedData[(combinedData$wavelength==wavelength1),]
    abs1<-abs1[,c("sampleID","absorbance")]
    names(abs1)[names(abs1) == "absorbance"] <- "abs1"
    abs2<-combinedData[(combinedData$wavelength==wavelength2),]
    abs2<-abs2[,c("sampleID","absorbance")]
    names(abs2)[names(abs2) == "absorbance"] <- "abs2"
    absRatio<-merge(abs1,abs2,by.x="sampleID",by.y="sampleID",all.x=T,all.y=T)
    absRatio$absRatio<-absRatio$abs1/absRatio$abs2


  # Performs calculation with Fe correction
  if(correctFe == TRUE){
    abs1Corrected<-combinedData[(combinedData$wavelength==wavelength1),]
    abs1Corrected<-abs1Corrected[,c("sampleID","absorbanceCorrected")]
    names(abs1Corrected)[names(abs1Corrected) == "absorbanceCorrected"] <- "abs1Corrected"
    abs2Corrected<-combinedData[(combinedData$wavelength==wavelength2),]
    abs2Corrected<-abs2Corrected[,c("sampleID","absorbanceCorrected")]
    names(abs2Corrected)[names(abs2Corrected) == "absorbanceCorrected"] <- "abs2Corrected"
    absRatioCorrected<-merge(abs1Corrected,abs2Corrected,by.x="sampleID",by.y="sampleID",all.x=T,all.y=T)
    absRatioCorrected$absRatioCorrected<-absRatioCorrected$abs1Corrected/absRatioCorrected$abs2Corrected
  }

  # Formats output table
  outputTable<-unique(absorbanceAveraged[,c("domainID","siteID","sampleID","collectDate")])
  outputTable<-merge(outputTable,absRatio,by.x="sampleID",by.y="sampleID",all.x=T,all.y=F)
  outputTable<-outputTable[,c("domainID","siteID","sampleID","collectDate","absRatio")]

  if(correctFe == TRUE){
    outputTable<-merge(outputTable,absRatioCorrected,by.x="sampleID",by.y="sampleID",all.x=T,all.y=F)
    outputTable<-outputTable[,c("domainID","siteID","sampleID","collectDate","absRatio","absRatioCorrected")]
  }

  return(outputTable)
}
