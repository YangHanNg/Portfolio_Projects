-- Useful query for pivoting financial metrics data
WITH financial_metrics_pivot AS (
    SELECT 
        c.symbol,
        rp.fiscal_date_ending,
        MAX(CASE WHEN m.metric_name = 'totalRevenue' THEN fd.value END) AS "totalRevenue",
        MAX(CASE WHEN m.metric_name = 'ebit' THEN fd.value END) AS ebit,
        MAX(CASE WHEN m.metric_name = 'researchAndDevelopment' THEN fd.value END) AS "researchAndDevelopment",
        MAX(CASE WHEN m.metric_name = 'interestExpense' THEN fd.value END) AS "interestExpense",
        MAX(CASE WHEN m.metric_name = 'incomeBeforeTax' THEN fd.value END) AS "incomeBeforeTax",
        MAX(CASE WHEN m.metric_name = 'incomeTaxExpense' THEN fd.value END) AS "incomeTaxExpense",
        MAX(CASE WHEN m.metric_name = 'cashAndShortTermInvestments' THEN fd.value END) AS "cashAndShortTermInvestments",
        MAX(CASE WHEN m.metric_name = 'totalCurrentAssets' THEN fd.value END) AS "totalCurrentAssets",
        MAX(CASE WHEN m.metric_name = 'totalCurrentLiabilities' THEN fd.value END) AS "totalCurrentLiabilities",
        MAX(CASE WHEN m.metric_name = 'totalAssets' THEN fd.value END) AS "totalAssets",
        MAX(CASE WHEN m.metric_name = 'totalLiabilities' THEN fd.value END) AS "totalLiabilities",
        MAX(CASE WHEN m.metric_name = 'depreciationAndAmortization' THEN fd.value END) AS "depreciationAndAmortization",
        MAX(CASE WHEN m.metric_name = 'capitalLeaseObligations' THEN fd.value END) AS "capitalLeaseObligations",
        MAX(CASE WHEN m.metric_name = 'longTermDebt' THEN fd.value END) AS "longTermDebt",
        MAX(CASE WHEN m.metric_name = 'currentLongTermDebt' THEN fd.value END) AS "currentLongTermDebt",
        MAX(CASE WHEN m.metric_name = 'propertyPlantEquipment' THEN fd.value END) AS "propertyPlantEquipment",
        ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY rp.fiscal_date_ending DESC) AS row_num
    FROM financial_data fd
    JOIN financial_metrics m ON fd.metric_id = m.metric_id
    JOIN financial_reports fr ON fd.report_id = fr.report_id
    JOIN reporting_periods rp ON fr.period_id = rp.period_id
    JOIN companies c ON fr.company_id = c.company_id
    WHERE 
        c.symbol = 'AAPL' AND 
        rp.period_type = 'Annual'
    GROUP BY c.symbol, rp.fiscal_date_ending
)
SELECT 
    symbol,
    fiscal_date_ending,
    "totalRevenue",
    ebit,
    "researchAndDevelopment",
    "interestExpense",
    "incomeBeforeTax",
    "incomeTaxExpense",
    "cashAndShortTermInvestments", 
    "totalCurrentAssets",
    "totalCurrentLiabilities",
    "totalAssets",
    "totalLiabilities",
    "depreciationAndAmortization",
    "capitalLeaseObligations",
    "longTermDebt",
    "currentLongTermDebt",
    "propertyPlantEquipment"
FROM financial_metrics_pivot
ORDER BY fiscal_date_ending DESC;