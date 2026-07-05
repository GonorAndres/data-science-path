# Mexican DS Job Boards: How to Use Them

Four boards matter for data science roles in Mexico in 2026. This guide covers how to search each, what filters actually work, expected recruiter response rates, and how to structure the daily application routine.

## The four boards

| Board | Primary use | Daily listings (DS in MX) | Recruiter response rate | Notes |
|-------|-------------|---------------------------|--------------------------|-------|
| LinkedIn MX | Multinational and tech | 40 to 80 | 5 to 15 percent | Highest leverage. Recruiter InMail works. |
| OCC Mundial | Broad MX market (banks, retail, consulting) | 20 to 40 | 3 to 8 percent | Largest MX-native board. ATS-heavy. |
| Indeed MX | Aggregator (crossposts) | 30 to 60 | 3 to 7 percent | Many duplicates. Filter by date. |
| Bumeran | LATAM roles, retail, SMB | 10 to 20 | 2 to 5 percent | Lower signal for senior DS. Useful for juniors. |

Beyond these four, two secondary channels:

- **Company careers pages** (highest conversion when used directly, not via aggregator). Citi, Nu, Bitso, Kavak, Coppel, Banorte, BBVA, Santander MX all have dedicated DS listings.
- **Twitter / X tech-MX community** (@empleoTI, @empleosTI, #EmpleoTI). Low volume, high trust signal when a hiring manager posts directly.

## LinkedIn MX

### Search query patterns

Use Boolean syntax in the search bar. Paste these as starting points:

```
("Data Scientist" OR "Data Science" OR "Cientifico de Datos" OR "ML Engineer" OR "Machine Learning Engineer") AND (Mexico OR CDMX OR "Ciudad de Mexico" OR remoto OR remote)
```

For junior roles specifically:

```
("Data Scientist" OR "Analista de Datos" OR "Jr Data") AND (junior OR "recien egresado" OR "trainee" OR "entry-level") AND (Mexico OR CDMX)
```

### Filter settings

- **Date posted**: last 24 hours for active watchlist; last 7 days for weekly sweep.
- **Experience level**: Entry level, Associate, Mid-Senior. Leave out "Director" unless applying aggressively up.
- **Job type**: Full-time and Internship checked.
- **Remote**: on-site, hybrid, remote all checked initially. Narrow later.
- **Location**: "Mexico City Metropolitan Area" + "Mexico" (country-level) for remote-friendly roles.

### Alerts

Create three saved searches with daily email:

1. CDMX + "Data Scientist"
2. Remote Mexico + "ML Engineer"
3. LATAM + "Senior Data Scientist" (optional if you are not junior)

### Application workflow

- Every application: quick tailor of the LinkedIn Easy Apply form (2 to 5 minutes).
- Every application: find one person at the target company and send a connection request with a one-line note. See `recruiter_outreach_templates.md`.
- Easy Apply alone converts poorly. Easy Apply + DM to a recruiter or hiring manager at the same company converts 3 to 5x better.

### Cadence

- 10 to 15 applications per weekday during active search.
- Follow up on each application 7 days later if no response.

## OCC Mundial

### Search tips

OCC search is less sophisticated than LinkedIn. Best query format:

```
data scientist
```

or

```
cientifico de datos
```

Do not use Boolean. OCC ignores it.

### Filters to actually set

- **Ciudad**: CDMX, Estado de Mexico, Jalisco (GDL), Nuevo Leon (MTY).
- **Modalidad**: Presencial, Hibrido, Home Office. Leave all checked.
- **Salario**: omit initially; salary filter hides many posts that do not list a range.
- **Experiencia**: 1 a 3 anos for junior; 3 a 5 for mid.

### Daily routine

- Log in every weekday morning.
- Check "Empleos recomendados" first. OCC's recommender is weak but catches a few.
- Apply via OCC's internal form. Upload the ES PDF (`AndresGonzalez_DataScientist_ES.pdf`).
- If the posting links to an external form (company careers page), always use the external link instead. OCC sometimes drops applications.

### Red flags on OCC

- Postings that demand "Excel avanzado" with no mention of Python or SQL. These are reporting roles, not DS.
- Postings that list 15+ technologies as "required". Usually consulting body-shop roles with low ML content.
- Postings with a salary under MXN 12k for a "senior" DS. Ignore.

## Indeed MX

Indeed is an aggregator. Most of its listings are reposts of LinkedIn or company careers pages. Use it for:

- Discovering employers you had not considered (scroll through the company logos).
- Catching postings that missed LinkedIn.

### Search tips

Indeed supports partial Boolean:

```
"data scientist" Mexico -manager -director
```

The `-word` pattern excludes unwanted results. Useful to remove noise.

### Filter settings

- Fecha: ultimas 24 horas.
- Tipo de empleo: tiempo completo.
- Ubicacion: CDMX + remoto.

### Warning

Indeed has more spam postings than OCC or LinkedIn. Verify the company exists via LinkedIn before applying. If the company has no LinkedIn page, skip.

## Bumeran

Bumeran is stronger for analyst and junior roles, weaker for senior DS. Good for casting a wide early-career net.

### Search

```
data scientist
```

or

```
machine learning
```

### Use cases

- Retail analytics roles (Liverpool, Walmart MX, Soriana post here).
- Startup and SMB DS roles.
- Fintech backoffice analytics.

Skip if you are targeting senior or specialized roles; LinkedIn and direct applications are higher conversion.

## Daily routine (active search)

| Time | Activity | Duration |
|------|----------|----------|
| 09:00 | Check LinkedIn alerts, apply to 5 new | 45 min |
| 10:00 | Check OCC, apply to 3 new | 30 min |
| 11:00 | Check Indeed and Bumeran, apply to 2 if worth it | 20 min |
| Later | Send 3 cold DMs to recruiters / hiring managers | 20 min |
| Later | Update tracking spreadsheet | 10 min |

Total: under 2 hours per day. Anything more burns out.

## Tracking

Maintain a spreadsheet with:

- Date applied
- Company
- Role
- Source (LinkedIn, OCC, referral, direct)
- Salary range if stated
- Response y/n + date
- Next step

Review weekly. Kill applications where no response after 14 days unless the company is a high priority.

## Expected numbers during a real search

Based on typical Mexican DS junior-to-mid experience in 2026:

- 100 applications in 30 days via mixed boards.
- 10 to 15 recruiter replies (first screen).
- 5 to 8 technical screens.
- 2 to 4 final rounds.
- 1 offer.

If the ratio is much lower, something upstream is broken: CV, LinkedIn, portfolio, or targeting. Revisit each before doubling volume.

## Anti-patterns

1. Posting "open to work" as a public banner. Signals desperation to recruiters.
2. Applying to the same company multiple times in a month. Wait 4 to 6 weeks between attempts.
3. Quantity over quality. 100 low-fit applications beat 30 tailored ones only if the 30 are genuinely tailored; they rarely are.
4. Skipping the DM step. Applications alone convert at 1 to 3 percent. Applications with a DM convert at 5 to 10 percent.
5. Ghosting after a technical screen went well. Follow up at 7 days if silent. Most silence is queue delay, not rejection.
