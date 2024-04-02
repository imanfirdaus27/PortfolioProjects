select *
from PortfolioProject..CovidDeaths
where continent is not null
order by 3,4

--select *
--from PortfolioProject..covidVaccinations
--order by 3,4

-- Select Data that we are going to be using

select Location, date, total_cases, new_cases, total_deaths, population
from PortfolioProject..CovidDeaths
where continent is not null
order by 1,2


-- Looking at Total cases vs Total deaths
-- Show likelihood of dying if yo contract covid in your country

select Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
from PortfolioProject..CovidDeaths
where location like '%states%'
where continent is not null
order by 1,2

--Looking at Total Cases vs Population
-- show what percentage of population got covid

select Location, date, Population, total_cases, (total_cases/population)*100 as PercentPopulationInfected
from PortfolioProject..CovidDeaths
--where location like '%States%'
where continent is not null
order by 1,2


-- Looking at Countries with Highest Infection Rate compared to Population

select Location, Population, Max(total_cases) as HighestInfectionCount, Max((total_cases/population))*100 as MaxPercentPopulationInfected
from PortfolioProject..CovidDeaths
--where location like '%States%'
where continent is not null
group by Location, Population
order by 4 desc


-- Showing Countries with Highest Death Count per Population

select Location, Max(cast(total_deaths as int)) as TotalDeathCount
from PortfolioProject..CovidDeaths
--where location like '%States%'
where continent is not null
group by location
order by 2 desc


-- LET'S BREAK THINGS DOWN BY CONTINENT


-- Showing continents with the highest death count

select continent, Max(cast(total_deaths as int)) as TotalDeathCount
from PortfolioProject..CovidDeaths
--where location like '%States%'
where continent is not null
group by continent
order by 2 desc


-- GLOBAL NUMBERS

select sum(new_cases) as total_cases, sum(cast(new_deaths as int)) as total_deaths, sum(cast(new_deaths as int))/sum(new_cases)*100 as DeathPercentage
from PortfolioProject..CovidDeaths
--where location like '%states%'
where continent is not null
--Group by date
order by 1,2


-- Looking at Total Poplualtion vs Vaccianations

select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (Partition by dea.Location order by dea.location, dea.date) as RollingPeopleVaccinated, --(RollingPeopleVaccinated/Population)*100
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
order by 2,3


-- USE CTE 

with PopvsVac (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated) as
(
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (Partition by dea.Location order by dea.location, dea.date) as RollingPeopleVaccinated --,(RollingPeopleVaccinated/Population)*100
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
--order by 2,3
)
Select *, (RollingPeopleVaccinated/Population)*100 as PercentRollingPeopleVaccinated
from PopvsVac



--USE TEMP TABLE

DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)


insert into #PercentPopulationVaccinated
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (Partition by dea.Location order by dea.location, dea.date) as RollingPeopleVaccinated --,(RollingPeopleVaccinated/Population)*100
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
--order by 2,3

Select *, (RollingPeopleVaccinated/Population)*100 as PercentRollingPeopleVaccinated
from #PercentPopulationVaccinated



-- Creating View to store data for later visualizations

Create view PercentPopulationVaccinated as 
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, sum(convert(int, vac.new_vaccinations)) over (Partition by dea.Location order by dea.location, dea.date) as RollingPeopleVaccinated --,(RollingPeopleVaccinated/Population)*100
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
--order by 2,3


Select *
From PercentPopulationVaccinated