def need_for_escalation(df_input): # : pd.DataFrame -> pd.DataFrame:
    """
    Determine if escalation is needed based on information in df.

    Args:
        df (pd.DataFrame): DataFrame containing level of 'expected_action', 'severity', 
        'uncertainty' and 'confidence'.
    """
    df = df_input.copy()

    def escalation_postprocess(
                        exp_action: bool, 
                        severity: str, 
                        confidence: float,
                        uncertainty_level: str,
                        n_risk_factors: int,
                        n_missing_information: int) -> bool:
        # default: trust the LLM
        # real_action = exp_action
        
        if exp_action is True:
            if severity == "low":
            # (
            # or confidence < 0.4
            # or uncertainty_level == "high"
            #     ):
                return False

        elif exp_action is False:
            if severity == "high":      # safety first
                return True
                
            if severity == "medium":
                if (confidence >= 0.8 or 
                    n_missing_information >= 2 or
                    n_risk_factors >= 2):

                    return True
                return False
                # or ()): # too less information
                # real_action = True
            
            # if () and (
            #         () or 
            #         ()):
            #     real_action = True


        return exp_action

    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    
    df["expected_action_final"] = df.apply(
                            lambda row: escalation_postprocess(
                                exp_action=row.get("expected_action_llm", "unknown"),
                                severity=row.get("severity", "unknown"), 
                                confidence=row.get("confidence", "unknown"), 
                                uncertainty_level=row.get("uncertainty_level", "unknown"), 
                                n_risk_factors=row.get("n_risk_factors", "unknown"),
                                n_missing_information=row.get("n_missing_information", "unknown")),
                                axis=1)

    event_logger.info("[CHECK] 'result_df (post-processed)' columns: \n%s\n", df.columns.tolist())

    return df
