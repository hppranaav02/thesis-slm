SELECT
  cc.cwe_id,
  fc.file_change_id,
  fc.code_before,
  fc.code_after,
  fc.diff,
  fc.filename,
  fc.change_type,
  c.msg,
  EXTRACT(YEAR FROM c.committer_date) AS commit_year
FROM
  cwe_classification AS cc
  JOIN fixes AS f
    ON cc.cve_id = f.cve_id
  JOIN commits AS c
    ON f.hash = c.hash
  JOIN file_change AS fc
    ON f.hash = fc.hash
WHERE
  fc.code_before IS NOT NULL
  AND fc.code_after  IS NOT NULL
  AND fc.programming_language = 'Go'
  AND cc.cwe_id <> 'NVD-CWE-noinfo'
  AND cc.cwe_id <> 'NVD-CWE-Other'
  AND fc.code_before <> 'None'
  AND fc.filename LIKE '%.go'
GROUP BY
    cc.cwe_id,
    fc.file_change_id,
    fc.code_before,
    fc.code_after,
    fc.diff,
    fc.filename,
    fc.change_type,
    c.msg,
    commit_year
ORDER BY
  fc.file_change_id;