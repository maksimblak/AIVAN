import React, { useState, useEffect } from 'react';
import {
  Scale, Search, FileText, User, Bell, Settings, Sun, Moon,
  ChevronRight, Shield, BookOpen, Briefcase, FileCheck, Sparkles,
  MessageSquare, Award, TrendingUp, Clock, Calendar, Star,
  Home, ArrowLeft, Plus, Download, Send, Mic, Image as ImageIcon,
  CheckCircle, AlertCircle, Info, X, Menu, Filter, BarChart3
} from 'lucide-react';

// ============ ГЛАВНЫЙ КОМПОНЕНТ ============
const LegalAIPro = () => {
  const [currentScreen, setCurrentScreen] = useState('main');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [notifications, setNotifications] = useState(3);
  const [userStats, setUserStats] = useState({
    questionsAsked: 47,
    documentsProcessed: 23,
    casesAnalyzed: 12,
    subscriptionDays: 145
  });

  // Переключение темы
  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  // Юридические категории из вашего бота
  const legalCategories = [
    { id: 'civil', name: 'Гражданское право', icon: '📘', color: 'blue' },
    { id: 'corporate', name: 'Корпоративное право', icon: '🏢', color: 'purple' },
    { id: 'contract', name: 'Договорное право', icon: '📝', color: 'indigo' },
    { id: 'labor', name: 'Трудовое право', icon: '⚙️', color: 'orange' },
    { id: 'tax', name: 'Налоговое право', icon: '💰', color: 'green' },
    { id: 'real_estate', name: 'Недвижимость', icon: '🏠', color: 'red' },
    { id: 'ip', name: 'Интеллектуальная собственность', icon: '🧠', color: 'pink' },
    { id: 'admin', name: 'Административное право', icon: '🏛️', color: 'gray' },
    { id: 'criminal', name: 'Уголовное право', icon: '🧑‍⚖️', color: 'rose' },
    { id: 'family', name: 'Семейное право', icon: '👪', color: 'teal' },
  ];

  // Базовые классы для темной темы
  const bg = isDarkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50';
  const cardBg = isDarkMode ? 'bg-gray-800' : 'bg-white/80 backdrop-blur-lg';
  const textPrimary = isDarkMode ? 'text-white' : 'text-gray-900';
  const textSecondary = isDarkMode ? 'text-gray-300' : 'text-gray-600';
  const borderColor = isDarkMode ? 'border-gray-700' : 'border-gray-200';

  // ============ КОМПОНЕНТЫ ЭКРАНОВ ============

  // ГЛАВНОЕ МЕНЮ
  const MainMenu = () => (
    <div className="flex flex-col h-full">
      {/* Хедер */}
      <div className={`${cardBg} rounded-2xl p-6 shadow-lg mb-4 border ${borderColor}`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Scale className="text-white" size={24} />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse"></div>
            </div>
            <div>
              <h1 className={`text-xl font-bold ${textPrimary}`}>LegalAI Pro</h1>
              <span className="text-xs bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-2 py-0.5 rounded-full font-semibold">
                Premium
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleTheme}
              className={`p-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl transition-all duration-300 hover:scale-110`}
            >
              {isDarkMode ? <Sun size={20} className="text-yellow-400" /> : <Moon size={20} className="text-gray-600" />}
            </button>
            <button className="relative p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl transition-all duration-300 hover:scale-110">
              <Bell size={20} className="text-white" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center font-bold">
                  {notifications}
                </span>
              )}
            </button>
            <button
              onClick={() => setCurrentScreen('settings')}
              className={`p-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl transition-all duration-300 hover:scale-110`}
            >
              <Settings size={20} className={textSecondary} />
            </button>
          </div>
        </div>

        {/* Статистика */}
        <div className="grid grid-cols-4 gap-3">
          <StatCard value={userStats.questionsAsked} label="Вопросов" icon={MessageSquare} />
          <StatCard value={userStats.documentsProcessed} label="Документов" icon={FileText} />
          <StatCard value={userStats.casesAnalyzed} label="Анализов" icon={BarChart3} />
          <StatCard value={userStats.subscriptionDays} label="Дней" icon={Star} />
        </div>
      </div>

      {/* Основные действия */}
      <div className="space-y-3 mb-4">
        <ActionCard
          title="Юридический вопрос"
          subtitle="Получите экспертную консультацию по любому правовому вопросу"
          icon={Scale}
          gradient="from-blue-600 to-blue-700"
          onClick={() => setCurrentScreen('legal-question')}
        />
        <ActionCard
          title="Поиск судебной практики"
          subtitle="Анализ дел и поиск релевантных решений судов"
          icon={Search}
          gradient="from-purple-600 to-purple-700"
          onClick={() => setCurrentScreen('search-practice')}
        />
        <ActionCard
          title="Работа с документами"
          subtitle="OCR, анализ рисков, составление и проверка документов"
          icon={FileText}
          gradient="from-pink-600 to-pink-700"
          onClick={() => setCurrentScreen('documents')}
        />
      </div>

      {/* Нижняя навигация */}
      <div className={`mt-auto ${cardBg} rounded-2xl shadow-lg border ${borderColor}`}>
        <div className="flex justify-around p-4">
          <NavButton icon={Home} label="Главная" active={true} />
          <NavButton icon={User} label="Профиль" onClick={() => setCurrentScreen('profile')} />
          <NavButton icon={MessageSquare} label="Поддержка" onClick={() => setCurrentScreen('support')} />
          <NavButton icon={Award} label="Подписка" />
        </div>
      </div>
    </div>
  );

  // ЭКРАН ЮРИДИЧЕСКОГО ВОПРОСА
  const LegalQuestionScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader title="Юридический вопрос" onBack={() => setCurrentScreen('main')} />

      <div className={`${cardBg} rounded-2xl p-5 shadow-lg mb-4 border ${borderColor}`}>
        <h3 className={`font-bold mb-3 ${textPrimary}`}>Выберите категорию права</h3>
        <div className="grid grid-cols-2 gap-3">
          {legalCategories.slice(0, 6).map((category) => (
            <CategoryCard
              key={category.id}
              category={category}
              onClick={() => {
                setSelectedCategory(category);
                setCurrentScreen('question-input');
              }}
            />
          ))}
        </div>
        <button className={`w-full mt-3 p-3 border-2 border-dashed ${borderColor} rounded-xl ${textSecondary} text-sm font-semibold hover:border-blue-500 transition-all duration-300`}>
          Показать все категории
        </button>
      </div>

      <QuickActions />
    </div>
  );

  // ЭКРАН ВВОДА ВОПРОСА
  const QuestionInputScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader
        title={selectedCategory?.name || "Ваш вопрос"}
        subtitle={selectedCategory?.icon}
        onBack={() => setCurrentScreen('legal-question')}
      />

      <div className={`flex-1 ${cardBg} rounded-2xl p-5 shadow-lg mb-4 border ${borderColor}`}>
        <div className="mb-4">
          <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
            Опишите вашу ситуацию
          </label>
          <textarea
            className={`w-full h-40 p-4 border ${borderColor} rounded-xl ${isDarkMode ? 'bg-gray-700 text-white' : 'bg-white'} resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300`}
            placeholder="Подробно опишите вашу правовую ситуацию. Чем детальнее описание, тем точнее будет консультация..."
          ></textarea>
        </div>

        <div className="grid grid-cols-3 gap-2 mb-4">
          <button className={`p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl flex flex-col items-center space-y-1 hover:scale-105 transition-all duration-300`}>
            <Mic size={20} className="text-blue-600" />
            <span className="text-xs text-gray-600">Голос</span>
          </button>
          <button className={`p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl flex flex-col items-center space-y-1 hover:scale-105 transition-all duration-300`}>
            <ImageIcon size={20} className="text-purple-600" />
            <span className="text-xs text-gray-600">Фото</span>
          </button>
          <button className={`p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl flex flex-col items-center space-y-1 hover:scale-105 transition-all duration-300`}>
            <FileText size={20} className="text-pink-600" />
            <span className="text-xs text-gray-600">Файл</span>
          </button>
        </div>

        <button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl font-bold flex items-center justify-center space-x-2 hover:shadow-2xl hover:scale-105 transition-all duration-300">
          <Sparkles size={20} />
          <span>Получить консультацию</span>
        </button>
      </div>

      <div className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor}`}>
        <div className="flex items-start space-x-3">
          <Info size={20} className="text-blue-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className={`text-sm ${textSecondary}`}>
              <span className="font-semibold">Совет:</span> Укажите даты, суммы, стороны договора и другие детали для более точной консультации.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  // ЭКРАН ПОИСКА ПРАКТИКИ
  const SearchPracticeScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader title="Поиск судебной практики" onBack={() => setCurrentScreen('main')} />

      <div className={`${cardBg} rounded-2xl p-4 shadow-lg mb-4 border ${borderColor}`}>
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Введите тему или номер дела..."
            className={`w-full pl-12 pr-12 py-4 border ${borderColor} rounded-xl ${isDarkMode ? 'bg-gray-700 text-white' : 'bg-white'} focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300`}
          />
          <button className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg">
            <Filter size={16} className="text-white" />
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto pb-4">
        <CaseCard
          title="Дело № А40-123456/2024"
          category="Гражданское право"
          court="Арбитражный суд г. Москвы"
          date="15 октября 2024"
          status="Решение в пользу истца"
          relevance={95}
        />
        <CaseCard
          title="Дело № А41-789012/2024"
          category="Корпоративное право"
          court="9 ААС"
          date="10 октября 2024"
          status="Апелляция отклонена"
          relevance={87}
        />
        <CaseCard
          title="Дело № А56-345678/2024"
          category="Договорное право"
          court="Арбитражный суд СПб"
          date="5 октября 2024"
          status="В процессе рассмотрения"
          relevance={78}
        />
      </div>
    </div>
  );

  // ЭКРАН РАБОТЫ С ДОКУМЕНТАМИ
  const DocumentsScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader title="Работа с документами" onBack={() => setCurrentScreen('main')} />

      <div className="grid grid-cols-2 gap-3 mb-4">
        <DocumentActionCard
          title="Загрузить документ"
          icon={Plus}
          color="blue"
        />
        <DocumentActionCard
          title="OCR сканирование"
          icon={ImageIcon}
          color="purple"
        />
        <DocumentActionCard
          title="Анализ рисков"
          icon={Shield}
          color="red"
        />
        <DocumentActionCard
          title="Составить документ"
          icon={FileCheck}
          color="green"
        />
      </div>

      <div className={`${cardBg} rounded-2xl p-4 shadow-lg mb-4 border ${borderColor}`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className={`font-bold ${textPrimary}`}>Недавние документы</h3>
          <button className="text-blue-600 text-sm font-semibold">Все</button>
        </div>
        <div className="space-y-2">
          <DocumentItem name="Договор поставки.pdf" date="Сегодня, 14:30" size="2.4 МБ" status="processed" />
          <DocumentItem name="Исковое заявление.docx" date="Вчера, 16:45" size="1.2 МБ" status="draft" />
          <DocumentItem name="Устав ООО.pdf" date="20 окт, 10:15" size="5.8 МБ" status="processed" />
        </div>
      </div>
    </div>
  );

  // ЭКРАН ПРОФИЛЯ
  const ProfileScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader title="Мой профиль" onBack={() => setCurrentScreen('main')} />

      <div className={`${cardBg} rounded-2xl p-6 shadow-lg mb-4 border ${borderColor}`}>
        <div className="flex items-center space-x-4 mb-6">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center text-white text-2xl font-bold">
            ЮР
          </div>
          <div className="flex-1">
            <h2 className={`text-xl font-bold ${textPrimary}`}>Юрий Романов</h2>
            <p className={`${textSecondary} text-sm`}>Юрист-консультант</p>
            <div className="flex items-center space-x-2 mt-2">
              <Award className="text-yellow-500" size={16} />
              <span className="text-sm font-semibold bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
                Premium подписка
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <ProfileStat
            icon={TrendingUp}
            value={`${userStats.questionsAsked}`}
            label="Вопросов задано"
            color="blue"
          />
          <ProfileStat
            icon={FileCheck}
            value={`${userStats.documentsProcessed}`}
            label="Документов"
            color="purple"
          />
          <ProfileStat
            icon={BarChart3}
            value={`${userStats.casesAnalyzed}`}
            label="Анализов"
            color="pink"
          />
          <ProfileStat
            icon={Calendar}
            value={`${userStats.subscriptionDays} дн.`}
            label="Подписка"
            color="green"
          />
        </div>
      </div>

      <div className="space-y-3">
        <ProfileMenuItem icon={Settings} title="Настройки" />
        <ProfileMenuItem icon={Bell} title="Уведомления" badge={notifications} />
        <ProfileMenuItem icon={Download} title="История запросов" />
        <ProfileMenuItem icon={MessageSquare} title="Служба поддержки" />
      </div>
    </div>
  );

  // ЭКРАН НАСТРОЕК
  const SettingsScreen = () => (
    <div className="flex flex-col h-full">
      <ScreenHeader title="Настройки" onBack={() => setCurrentScreen('main')} />

      <div className="space-y-4">
        <SettingsSection title="Внешний вид">
          <SettingToggle
            label="Темная тема"
            checked={isDarkMode}
            onChange={toggleTheme}
            icon={isDarkMode ? Moon : Sun}
          />
          <SettingToggle
            label="Компактный режим"
            checked={false}
          />
        </SettingsSection>

        <SettingsSection title="Уведомления">
          <SettingToggle
            label="Push-уведомления"
            checked={true}
            icon={Bell}
          />
          <SettingToggle
            label="Email-рассылка"
            checked={false}
          />
        </SettingsSection>

        <SettingsSection title="Приватность">
          <SettingToggle
            label="Аналитика использования"
            checked={true}
            icon={BarChart3}
          />
          <SettingToggle
            label="История запросов"
            checked={true}
          />
        </SettingsSection>

        <div className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor}`}>
          <button className="w-full text-red-600 font-semibold py-3 hover:bg-red-50 rounded-lg transition-all duration-300">
            Выйти из аккаунта
          </button>
        </div>
      </div>
    </div>
  );

  // ============ ВСПОМОГАТЕЛЬНЫЕ КОМПОНЕНТЫ ============

  const ScreenHeader = ({ title, subtitle, onBack }) => (
    <div className={`${cardBg} rounded-2xl p-4 shadow-lg mb-4 border ${borderColor}`}>
      <div className="flex items-center space-x-3">
        <button
          onClick={onBack}
          className={`p-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl hover:scale-110 transition-all duration-300`}
        >
          <ArrowLeft size={20} className={textPrimary} />
        </button>
        <div className="flex-1">
          <h2 className={`text-lg font-bold ${textPrimary}`}>{title}</h2>
          {subtitle && <span className="text-2xl">{subtitle}</span>}
        </div>
      </div>
    </div>
  );

  const StatCard = ({ value, label, icon: Icon }) => (
    <div className={`${isDarkMode ? 'bg-gray-700/50' : 'bg-gradient-to-br from-blue-50 to-purple-50'} rounded-xl p-3 text-center`}>
      <Icon size={16} className="mx-auto mb-1 text-blue-600" />
      <div className={`text-lg font-bold ${textPrimary}`}>{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );

  const ActionCard = ({ title, subtitle, icon: Icon, gradient, onClick }) => (
    <button
      onClick={onClick}
      className={`${cardBg} rounded-2xl p-5 shadow-lg border ${borderColor} w-full text-left hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 group`}
    >
      <div className="flex items-center space-x-4">
        <div className={`w-14 h-14 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center group-hover:scale-110 transition-all duration-300`}>
          <Icon className="text-white" size={24} />
        </div>
        <div className="flex-1">
          <h3 className={`font-bold ${textPrimary} mb-1`}>{title}</h3>
          <p className={`text-sm ${textSecondary}`}>{subtitle}</p>
        </div>
        <ChevronRight className={`${textSecondary} group-hover:translate-x-1 transition-all duration-300`} size={20} />
      </div>
    </button>
  );

  const CategoryCard = ({ category, onClick }) => {
    const colorClasses = {
      blue: 'from-blue-500 to-blue-600',
      purple: 'from-purple-500 to-purple-600',
      indigo: 'from-indigo-500 to-indigo-600',
      orange: 'from-orange-500 to-orange-600',
      green: 'from-green-500 to-green-600',
      red: 'from-red-500 to-red-600',
      pink: 'from-pink-500 to-pink-600',
      gray: 'from-gray-500 to-gray-600',
      rose: 'from-rose-500 to-rose-600',
      teal: 'from-teal-500 to-teal-600'
    };

    return (
      <button
        onClick={onClick}
        className={`${cardBg} rounded-xl p-4 border ${borderColor} hover:shadow-xl hover:scale-105 transition-all duration-300`}
      >
        <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[category.color]} rounded-lg flex items-center justify-center text-2xl mb-2`}>
          {category.icon}
        </div>
        <h4 className={`font-semibold text-sm ${textPrimary}`}>{category.name}</h4>
      </button>
    );
  };

  const NavButton = ({ icon: Icon, label, active, onClick }) => (
    <button
      onClick={onClick}
      className={`flex flex-col items-center space-y-1 transition-all duration-300 ${active ? 'scale-110' : 'opacity-60 hover:opacity-100'}`}
    >
      <div className={`p-2 rounded-xl ${active ? 'bg-gradient-to-br from-blue-600 to-purple-600' : isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
        <Icon size={20} className={active ? 'text-white' : textSecondary} />
      </div>
      <span className={`text-xs font-semibold ${active ? 'text-blue-600' : textSecondary}`}>{label}</span>
    </button>
  );

  const QuickActions = () => (
    <div className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor}`}>
      <h3 className={`font-bold mb-3 ${textPrimary}`}>Быстрые действия</h3>
      <div className="grid grid-cols-3 gap-2">
        <QuickActionButton icon={Mic} label="Голосовой" color="blue" />
        <QuickActionButton icon={FileText} label="Шаблоны" color="purple" />
        <QuickActionButton icon={Clock} label="История" color="pink" />
      </div>
    </div>
  );

  const QuickActionButton = ({ icon: Icon, label, color }) => {
    const colorClasses = {
      blue: 'from-blue-500 to-blue-600',
      purple: 'from-purple-500 to-purple-600',
      pink: 'from-pink-500 to-pink-600'
    };

    return (
      <button className={`p-3 bg-gradient-to-br ${colorClasses[color]} rounded-xl flex flex-col items-center space-y-1 hover:scale-105 transition-all duration-300`}>
        <Icon size={20} className="text-white" />
        <span className="text-xs text-white font-semibold">{label}</span>
      </button>
    );
  };

  const CaseCard = ({ title, category, court, date, status, relevance }) => (
    <div className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor} hover:shadow-2xl transition-all duration-300`}>
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <h4 className={`font-bold ${textPrimary} mb-1`}>{title}</h4>
          <p className={`text-sm ${textSecondary}`}>{category}</p>
        </div>
        <div className="bg-gradient-to-r from-green-500 to-green-600 text-white px-3 py-1 rounded-full text-xs font-bold">
          {relevance}%
        </div>
      </div>
      <div className={`flex items-center space-x-2 text-sm ${textSecondary} mb-2`}>
        <Briefcase size={14} />
        <span>{court}</span>
      </div>
      <div className={`flex items-center justify-between text-sm ${textSecondary}`}>
        <div className="flex items-center space-x-2">
          <Calendar size={14} />
          <span>{date}</span>
        </div>
        <span className="font-semibold text-blue-600">{status}</span>
      </div>
    </div>
  );

  const DocumentActionCard = ({ title, icon: Icon, color }) => {
    const colorClasses = {
      blue: 'from-blue-500 to-blue-600',
      purple: 'from-purple-500 to-purple-600',
      red: 'from-red-500 to-red-600',
      green: 'from-green-500 to-green-600'
    };

    return (
      <button className={`${cardBg} rounded-2xl p-4 border ${borderColor} hover:shadow-xl hover:scale-105 transition-all duration-300`}>
        <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[color]} rounded-xl flex items-center justify-center mb-3`}>
          <Icon className="text-white" size={24} />
        </div>
        <h4 className={`font-semibold text-sm ${textPrimary}`}>{title}</h4>
      </button>
    );
  };

  const DocumentItem = ({ name, date, size, status }) => {
    const statusConfig = {
      processed: { icon: CheckCircle, color: 'text-green-500', label: 'Обработан' },
      draft: { icon: Clock, color: 'text-orange-500', label: 'Черновик' }
    };

    const config = statusConfig[status];
    const Icon = config.icon;

    return (
      <div className={`flex items-center space-x-3 p-3 ${isDarkMode ? 'bg-gray-700/50' : 'bg-gray-50'} rounded-xl hover:bg-blue-50 transition-all duration-300`}>
        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
          <FileText className="text-white" size={20} />
        </div>
        <div className="flex-1">
          <h5 className={`font-semibold text-sm ${textPrimary}`}>{name}</h5>
          <p className="text-xs text-gray-500">{date} • {size}</p>
        </div>
        <Icon className={config.color} size={20} />
      </div>
    );
  };

  const ProfileStat = ({ icon: Icon, value, label, color }) => {
    const colorClasses = {
      blue: 'from-blue-500 to-blue-600',
      purple: 'from-purple-500 to-purple-600',
      pink: 'from-pink-500 to-pink-600',
      green: 'from-green-500 to-green-600'
    };

    return (
      <div className={`${isDarkMode ? 'bg-gray-700/50' : 'bg-gradient-to-br from-blue-50 to-purple-50'} rounded-xl p-4`}>
        <div className={`w-10 h-10 bg-gradient-to-br ${colorClasses[color]} rounded-lg flex items-center justify-center mb-2`}>
          <Icon className="text-white" size={20} />
        </div>
        <div className={`text-xl font-bold ${textPrimary}`}>{value}</div>
        <div className="text-xs text-gray-500">{label}</div>
      </div>
    );
  };

  const ProfileMenuItem = ({ icon: Icon, title, badge }) => (
    <button className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor} w-full hover:shadow-xl hover:-translate-y-1 transition-all duration-300`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-xl`}>
            <Icon size={20} className={textPrimary} />
          </div>
          <span className={`font-semibold ${textPrimary}`}>{title}</span>
        </div>
        <div className="flex items-center space-x-2">
          {badge && (
            <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full font-bold">
              {badge}
            </span>
          )}
          <ChevronRight className={textSecondary} size={20} />
        </div>
      </div>
    </button>
  );

  const SettingsSection = ({ title, children }) => (
    <div className={`${cardBg} rounded-2xl p-4 shadow-lg border ${borderColor}`}>
      <h3 className={`font-bold mb-3 ${textPrimary}`}>{title}</h3>
      <div className="space-y-3">
        {children}
      </div>
    </div>
  );

  const SettingToggle = ({ label, checked, onChange, icon: Icon }) => (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-3">
        {Icon && (
          <div className={`p-2 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg`}>
            <Icon size={16} className={textPrimary} />
          </div>
        )}
        <span className={`${textPrimary} font-medium`}>{label}</span>
      </div>
      <button
        onClick={onChange}
        className={`w-12 h-6 rounded-full transition-all duration-300 ${checked ? 'bg-gradient-to-r from-blue-600 to-purple-600' : 'bg-gray-300'}`}
      >
        <div className={`w-5 h-5 bg-white rounded-full shadow-lg transition-all duration-300 ${checked ? 'translate-x-6' : 'translate-x-1'}`}></div>
      </button>
    </div>
  );

  // ============ РОУТИНГ ЭКРАНОВ ============
  const renderScreen = () => {
    switch (currentScreen) {
      case 'main':
        return <MainMenu />;
      case 'legal-question':
        return <LegalQuestionScreen />;
      case 'question-input':
        return <QuestionInputScreen />;
      case 'search-practice':
        return <SearchPracticeScreen />;
      case 'documents':
        return <DocumentsScreen />;
      case 'profile':
        return <ProfileScreen />;
      case 'settings':
        return <SettingsScreen />;
      default:
        return <MainMenu />;
    }
  };

  // ============ ОСНОВНОЙ РЕНДЕР ============
  return (
    <div className={`min-h-screen ${bg} p-4 transition-colors duration-500`}>
      <div className="max-w-md mx-auto h-screen">
        {renderScreen()}
      </div>
    </div>
  );
};

export default LegalAIPro;
