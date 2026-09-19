WITH sent AS (
  SELECT
    CONCAT(From_Bank, '_', From_Account) AS account_key,
    CONCAT(To_Bank, '_', To_Account) AS counterparty_key,
    Timestamp,
    Amount_Paid AS montant,
    Payment_Currency AS devise,
    To_Bank AS contrepartie_bank,
    Payment_Format AS format_paiement,
    Is_Laundering
  FROM aml_detection.transactions_ibm
),
received AS (
  SELECT
    CONCAT(To_Bank, '_', To_Account) AS account_key,
    CONCAT(From_Bank, '_', From_Account) AS counterparty_key,
    Timestamp,
    Amount_Received AS montant,
    Receiving_Currency AS devise,
    From_Bank AS contrepartie_bank,
    Payment_Format AS format_paiement,
    Is_Laundering
  FROM aml_detection.transactions_ibm
),
sent_agg AS (
  SELECT
    account_key,
    COUNT(*) AS nb_transactions_envoyees,
    SUM(montant) AS montant_total_envoye,
    AVG(montant) AS montant_moyen_envoye,
    MAX(montant) AS montant_max_envoye,
    COUNT(DISTINCT devise) AS nb_devises_envoi,
    COUNT(DISTINCT contrepartie_bank) AS nb_banques_destinataires,
    COUNT(DISTINCT counterparty_key) AS nb_contreparties_envoi,
    COUNT(DISTINCT format_paiement) AS nb_formats_paiement,
    MAX(Is_Laundering) AS label_envoi
  FROM sent
  GROUP BY account_key
),
received_agg AS (
  SELECT
    account_key,
    COUNT(*) AS nb_transactions_recues,
    SUM(montant) AS montant_total_recu,
    AVG(montant) AS montant_moyen_recu,
    MAX(montant) AS montant_max_recu,
    COUNT(DISTINCT devise) AS nb_devises_reception,
    COUNT(DISTINCT contrepartie_bank) AS nb_banques_expediteurs,
    COUNT(DISTINCT counterparty_key) AS nb_contreparties_reception,
    MAX(Is_Laundering) AS label_reception
  FROM received
  GROUP BY account_key
),
velocity AS (
  SELECT account_key, MAX(nb_par_jour) AS max_transactions_jour
  FROM (
    SELECT
      account_key,
      DATE(PARSE_TIMESTAMP('%Y/%m/%d %H:%M', Timestamp)) AS jour,
      COUNT(*) AS nb_par_jour
    FROM sent
    GROUP BY account_key, jour
  )
  GROUP BY account_key
),
-- Réciprocité : un compte qui reçoit ET envoie vers la même contrepartie
-- est un signal classique de layering (flux circulaires)
reciprocite AS (
  SELECT
    s.account_key,
    COUNT(DISTINCT s.counterparty_key) AS nb_contreparties_reciproques
  FROM sent s
  INNER JOIN received r
    ON s.account_key = r.account_key
    AND s.counterparty_key = r.counterparty_key
  GROUP BY s.account_key
)
SELECT
  COALESCE(s.account_key, r.account_key) AS account_key,
  IFNULL(s.nb_transactions_envoyees, 0) AS nb_transactions_envoyees,
  IFNULL(r.nb_transactions_recues, 0) AS nb_transactions_recues,
  IFNULL(s.montant_total_envoye, 0) AS montant_total_envoye,
  IFNULL(r.montant_total_recu, 0) AS montant_total_recu,
  IFNULL(s.montant_moyen_envoye, 0) AS montant_moyen_envoye,
  IFNULL(r.montant_moyen_recu, 0) AS montant_moyen_recu,
  IFNULL(s.montant_max_envoye, 0) AS montant_max_envoye,
  IFNULL(r.montant_max_recu, 0) AS montant_max_recu,
  IFNULL(s.nb_devises_envoi, 0) AS nb_devises_envoi,
  IFNULL(r.nb_devises_reception, 0) AS nb_devises_reception,
  IFNULL(s.nb_banques_destinataires, 0) AS nb_banques_destinataires,
  IFNULL(r.nb_banques_expediteurs, 0) AS nb_banques_expediteurs,
  IFNULL(s.nb_formats_paiement, 0) AS nb_formats_paiement,
  IFNULL(v.max_transactions_jour, 0) AS max_transactions_jour,
  SAFE_DIVIDE(IFNULL(r.montant_total_recu, 0), NULLIF(IFNULL(s.montant_total_envoye, 0), 0)) AS ratio_recu_envoye,
  -- Nouvelles features de graphe
  IFNULL(s.nb_contreparties_envoi, 0) AS nb_contreparties_envoi,
  IFNULL(r.nb_contreparties_reception, 0) AS nb_contreparties_reception,
  IFNULL(rec.nb_contreparties_reciproques, 0) AS nb_contreparties_reciproques,
  SAFE_DIVIDE(
    IFNULL(rec.nb_contreparties_reciproques, 0),
    NULLIF(IFNULL(s.nb_contreparties_envoi, 0) + IFNULL(r.nb_contreparties_reception, 0), 0)
  ) AS ratio_reciprocite,
  GREATEST(IFNULL(s.label_envoi, 0), IFNULL(r.label_reception, 0)) AS label_laundering
FROM sent_agg s
FULL OUTER JOIN received_agg r ON s.account_key = r.account_key
LEFT JOIN velocity v ON COALESCE(s.account_key, r.account_key) = v.account_key
LEFT JOIN reciprocite rec ON COALESCE(s.account_key, r.account_key) = rec.account_key
