import pretty_midi as pm
from pathlib import Path
import sys
import numpy as np
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mido

import midi_functions  as mf
import config

# グローバル変数
# config.HIGH_CROP = 84 #where to cut off high notes. 84 = C6
# config.LOW_CROP = 24 #where to cut off low notes. 24=C1

# config.INSTRUMENT_ATTACH_METHOD = '1hot-category'
# config.SAVE_PREROCESSED_MIDI = False

name = 'preprocessed_midi'

# config.INPUT_LENGTH = 16
# config.OUTPUT_LENGTH = 16
# config.MAX_VOICES = 4
# config.MAX_SONGS = 5000

print_anything = False

def load_rolls(path, smallest_note = 16, maximal_number_of_voices_per_track = 1, MAX_VOICES = 4, include_only_monophonic_instruments=False, include_silent_note = True, velocity_threshold_such_that_it_is_a_played_note = 0.5, max_velocity = 127):
    # MIDIをロード
    # 失敗した場合はNoneを返す
    try:
        mid = pm.PrettyMIDI(path)
    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
        exception_str = 'Unexpected error in ' + path + ':\n', e, sys.exc_info()[0]
        if print_anything:print(exception_str)
        return None, None, None, None, None, None

    if print_anything:print("Time signature changes: ", mid.time_signature_changes)

    # 曲の始まりと終わりを決定
    # 途中でテンポが変わる場合は、テンポが安定している最も長い箇所(以降、「選択箇所」と呼びます)を選択
    # サイレント部分はカット
    # 小節の開始地点が曲中で一致しているか確認する
    tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
    song_start = 0
    song_end = mid.get_end_time()

    if len(tempo_change_times) > 1:
        longest_part = 0
        longest_part_start_time = 0
        longest_part_end_time = song_end
        longest_part_tempo = 0

        for i, tempo_change_time in enumerate(tempo_change_times):
            if i == len(tempo_change_times) - 1:
                end_time = song_end
            else:
                end_time = tempo_change_times[i+1]
            current_part_length = end_time - tempo_change_time
            if current_part_length > longest_part:
                longest_part = current_part_length
                longest_part_start_time = tempo_change_time
                longest_part_end_time = end_time
                longest_part_tempo = tempo_change_bpm[i]

        song_start = longest_part_start_time
        song_end = longest_part_end_time
        tempo = longest_part_tempo
    else:
        tempo = tempo_change_bpm[0]

    # 選択箇所以外の音を切る
    for instrument in mid.instruments:
        new_notes = [] # 残す音を入れるリスト
        for note in instrument.notes:
            # 選択箇所ないに入っているかチェック
            if note.start >= song_start and note.end <= song_end:
                # 時刻を選択箇所内に揃える
                note.start -= song_start
                note.end -= song_start
                new_notes.append(note)
        instrument.notes = new_notes

    # 曲内の音数によってピアノロールを降順に並べる
    number_of_notes = []
    piano_rolls = [i.get_piano_roll(fs=100) for i in mid.instruments]
    for piano_roll in piano_rolls:
        number_of_notes.append(np.count_nonzero(piano_roll))
    permutation = np.argsort(number_of_notes)[::-1]
    mid.instruments = [mid.instruments[i] for i in permutation]

    if print_anything:print("Song start(s):", song_start)
    if print_anything:print("Song end(s):", song_end)
    if print_anything:print("Tempo(bpm):", tempo)

    quarter_note_length = 60. / tempo
    # fs：サンプリング周波数
    # quarter_note_length * 4.：1小節の長さ(s)
    # smallest_noteは最小音符(e.g. 16分音符ならsmallest_note = 16)
    fs = 1./(quarter_note_length * 4./ smallest_note)
    total_ticks = math.ceil(song_end*fs)

    if print_anything:print("fs:", fs)
    if print_anything:print("Total ticks", total_ticks)

    # piano_rolls, velocity_rolls, note_rollsの組み立て
    piano_rolls = []
    velocity_rolls = []
    held_note_rolls = []
    max_concurrent_notes_per_track_list = []
    for instrument in mid.instruments:
        piano_roll = np.zeros((total_ticks, 128))

        # 同時発音数のためのリスト
        concurrent_notes_count = np.zeros((total_ticks,))

        # key:タプル(tick_start_of_the_note, pitch)
        # これは必ずしも一意とは限らない
        note_to_velocity_dict = dict()

        # key:タプル(tick_start_of_the_note, pitch)
        # これは必ずしも一意とは限らない
        note_to_duration_dict = dict()

        for note in instrument.notes:
            note_tick_start = note.start * fs
            note_tick_end = note.end * fs
            absolute_start = int(round(note_tick_start))
            absolute_end = int(round(note_tick_end))
            decimal = note_tick_start -absolute_start

            # tick付近の音or1s以上の音ならばpiano_rollに追加(このため実際の音楽と少し異なる)
            if decimal < 10e-3 or absolute_end-absolute_start >= 1:
                piano_roll[absolute_start:absolute_end, note.pitch] = 1
                concurrent_notes_count[absolute_start:absolute_end] += 1

                note_to_velocity_dict[(absolute_start, note.pitch)] = note.velocity
                note_to_duration_dict[(absolute_start, note.pitch)] = absolute_end - absolute_start

        max_concurrent_notes = int(np.max(concurrent_notes_count))
        max_concurrent_notes_per_track_list.append(max_concurrent_notes)
        if print_anything:print("program:", instrument.program)
        if print_anything:print("Max concurrent notes:", max_concurrent_notes)

        velocity_roll = np.zeros((total_ticks, max_concurrent_notes))
        held_note_roll = np.zeros((total_ticks, max_concurrent_notes))

        # step:時刻(tick)、note_vector:128次元のベクトルで各要素は音高に相当．その音があれば1となる．
        for step, note_vector in enumerate(piano_roll):
            # pitches:鳴っている音のid(たぶんnote_vectorのインデックスに相当)をまとめたベクトル
            pitches = list(note_vector.nonzero()[0])
            sorted_pitches_from_highest_to_lowest = sorted(pitches)[::-1]
            for voice_number, pitch in enumerate(sorted_pitches_from_highest_to_lowest):
                if (step, pitch) in note_to_velocity_dict.keys():
                    velocity_roll[step, voice_number] = note_to_velocity_dict[(step, pitch)]
                if (step, pitch) not in note_to_duration_dict.keys():
                    held_note_roll[step, voice_number] = 1

        piano_rolls.append(piano_roll)
        velocity_rolls.append(velocity_roll)
        held_note_rolls.append(held_note_roll)

    # 各楽器のprogram numberを入手 
    # program number:0~127の楽器と一対一で対応する番号
    programs = [i.program for i in mid.instruments]

    if print_anything:print(max_concurrent_notes_per_track_list)
    # 事前に設定した最大発音数を超えないようにするためのリスト
    override_max_notes_per_track_list = [maximal_number_of_voices_per_track for _ in max_concurrent_notes_per_track_list] 
    # トラック数をmax_voice個に制限したときに生じる無音のトラック数(そもそも元々のトラック数がmax_voice未満の場合、無音のトラックが生じる)
    silent_tracks_if_we_dont_override = config.MAX_VOICES - sum([min(maximal_number_of_voices_per_track, x) if x > 0 else 0 for x in max_concurrent_notes_per_track_list[:config.MAX_VOICES]])

    if print_anything:print("Silent tracks if no override:", silent_tracks_if_we_dont_override)
    # 「silent_tracks_if_we_dont_overrideが存在andあるトラックの最大発音数が設定よりオーバーしている」とき
    # ->silent_tracks_if_we_dont_override分だけオーバーしているトラックの最大発音数の上限を上げることができる
    for voice in range(min(config.MAX_VOICES, len(max_concurrent_notes_per_track_list))):
        if silent_tracks_if_we_dont_override > 0 and max_concurrent_notes_per_track_list[voice] > maximal_number_of_voices_per_track:
            additional_voices = min(silent_tracks_if_we_dont_override, max_concurrent_notes_per_track_list[voice]-maximal_number_of_voices_per_track)
            override_max_notes_per_track_list[voice] += additional_voices
            silent_tracks_if_we_dont_override -= additional_voices
    if print_anything:print("Override programs:", override_max_notes_per_track_list)

    # 発音数を最大同時発音数以内にするためにpiano_rollを選択する
    chosen_piano_rolls = []
    chosen_velocity_rolls = []
    chosen_held_note_rolls = []
    chosen_programs = []
    max_song_length = 0

    for piano_roll, velocity_roll, held_note_roll, program, max_concurrent_notes, override_max_notes_per_track in zip(piano_rolls, velocity_rolls, held_note_rolls, programs, max_concurrent_notes_per_track_list, override_max_notes_per_track_list):
        if max_concurrent_notes > 0:
            if include_only_monophonic_instruments:
                if max_concurrent_notes > 1:
                    if print_anything:print("Skipping this piano roll since it's polyphonic. Prigram number ", program)
                    continue
                else:
                    pass
                    if print_anything:print("Adding monophonic program number: ", program)
                
                monophonic_piano_roll = piano_roll

                if len(chosen_piano_rolls) < config.MAX_VOICES:
                    chosen_piano_rolls.append(monophonic_piano_roll)
                    chosen_velocity_rolls.append(velocity_roll)
                    chosen_held_note_rolls.append()
                    chosen_programs.append(program)
                    if monophonic_piano_roll.shape[0] > max_song_length:
                        max_song_length = monophonic_piano_roll.shape[0]
                    else:
                        break
            else: # 初期設定ではこちら
                # トラックあたりのボイス数を、トラックあたりの実際の同時使用ボイスの最小値、または設定ファイルで許可されている最大値で制限
                for voice in range(min(max_concurrent_notes, max(maximal_number_of_voices_per_track, override_max_notes_per_track))):
                    # 最高音をvoice 0, その次をvoice 1, ...とする
                    monophonic_piano_roll = np.zeros(piano_roll.shape)
                    for step in range(piano_roll.shape[0]):
                        # 音の高い順に並び替え
                        notes = np.nonzero(piano_roll[step, :])[0][::-1]
                        if len(notes) > voice:
                            monophonic_piano_roll[step, notes[voice]] = 1
                    
                    # これらをchosenに追加する
                    if len(chosen_piano_rolls) < config.MAX_VOICES:
                        chosen_piano_rolls.append(monophonic_piano_roll)
                        chosen_velocity_rolls.append(velocity_roll[:, voice])
                        chosen_held_note_rolls.append(held_note_roll[:, voice])
                        chosen_programs.append(program)
                        if monophonic_piano_roll.shape[0] > max_song_length:
                            max_song_length = monophonic_piano_roll.shape[0]
                    else:
                        break
                if len(chosen_piano_rolls) == config.MAX_VOICES:
                    break
    
    assert(len(chosen_piano_rolls) == len(chosen_velocity_rolls))
    assert(len(chosen_piano_rolls) == len(chosen_held_note_rolls))
    assert(len(chosen_piano_rolls) == len(chosen_programs))

    if len(chosen_piano_rolls) > 0:
        song_length = max_song_length*config.MAX_VOICES

        # Yの準備
        # Y:ターゲットノート
        Y = np.zeros((song_length, chosen_piano_rolls[0].shape[1]))
        # テンソルとなっているピアノロールを展開して一つのマトリックスに
        for i, piano_roll in enumerate(chosen_piano_rolls):
            for step in range(piano_roll.shape[0]):
                Y[i + step*config.MAX_VOICES, :] += piano_roll[step, :]
        
        for step in range(Y.shape[0]):
            # 演奏される音は最大でも1音
            assert(np.sum(Y[step, :]) <= 1)
        # 想定する音高の範囲で切り取る
        # これにより特徴空間を大幅に削減できる
        Y = Y[:, config.LOW_CROP:config.HIGH_CROP]
        # 無音を追加
        if include_silent_note:
            Y = np.append(Y, np.zeros((Y.shape[0], 1)), axis=1)
            for step in range(Y.shape[0]):
                if np.sum(Y[step]) == 0:
                    Y[step, -1] = 1
            # 各ステップのどこかには1があるようにする
            for step in range(Y.shape[0]):
                assert(np.sum(Y[step, :] == 1))

        # velocity rollの展開
        V = np.zeros((song_length,))
        for i, velocity_roll in enumerate(chosen_velocity_rolls):
            for step in range(velocity_roll.shape[0]):
                if velocity_roll[step] > 0:
                    velocity = velocity_threshold_such_that_it_is_a_played_note + (velocity_roll[step]/max_velocity) * (1.0 - velocity_threshold_such_that_it_is_a_played_note)
                    # 音量は最低でも 0.1*max_velocity??
                    # でも無音と演奏音の違いははっきりするらしい
                    assert(velocity <= 1.0)
                    V[i + step*config.MAX_VOICES] = velocity  
                
        # held_note_rollの展開
        # D:shapeは((song_length,)), {0,1}を格納しており1は継続, DurationのD
        D = np.zeros((song_length,)) 
        for i, held_note_roll in enumerate(chosen_held_note_rolls):
            for step in range(held_note_roll.shape[0]):
                D[i + step*config.MAX_VOICES] = held_note_roll[step]

        instrument_feature_matrix = mf.programs_to_instrument_matrix(chosen_programs, config.INSTRUMENT_ATTACH_METHOD, config.MAX_VOICES)
        # attach_instruments = falseなので省略
        # song_completion = falseなので省略

        X = Y

        if config.SAVE_PREROCESSED_MIDI: mf.rolls_to_midi(Y,chosen_programs, 'preprccced_data/', name, tempo, V, D)

        # 曲をoutput_length or input_lengthのサイズのチャンクに分ける
        # 必要に応じて無音を追加
        # paddingは右側に？
        if config.INPUT_LENGTH > 0:
            # Xを分割
            padding_length = config.INPUT_LENGTH - (X.shape[0] % config.INPUT_LENGTH) # input_lengthの長さに分割したときの不足している長さ
            if config.INPUT_LENGTH == padding_length:
                padding_length = 0
            X = np.pad(X, ((0, padding_length), (0,0)), 'constant', constant_values = (0,0)) # padding_length分だけパディング
            if include_silent_note:
                X[-padding_length:, -1] = 1 # パディングした部分は無音なので、無音のカラムに1を入れる
            number_of_splits = X.shape[0] // config.INPUT_LENGTH # 分割した数
            X = np.split(X, number_of_splits)
            X = np.asarray(X)

        if config.OUTPUT_LENGTH > 0:
            # Yを分割
            padding_length = config.OUTPUT_LENGTH - (Y.shape[0] % config.OUTPUT_LENGTH)
            if config.OUTPUT_LENGTH == padding_length:
                padding_length = 0
            
            Y = np.pad(Y, ((0, padding_length),(0,0)), 'constant', constant_values=(0,0))
            if include_silent_note:
                Y[-padding_length:, -1] = 1
            number_of_splits = Y.shape[0] // config.OUTPUT_LENGTH
            Y = np.split(Y, number_of_splits)
            Y = np.asarray(Y)

            # Vの分割
            V = np.pad(V, (0,padding_length), 'constant', constant_values=0)
            number_of_splits = V.shape[0] // config.OUTPUT_LENGTH
            V = np.split(V, number_of_splits)
            V = np.asarray(V)

            # Dの分割
            D = np.pad(D, (0,padding_length), 'constant', constant_values=0)
            number_of_splits = D.shape[0] // config.OUTPUT_LENGTH
            D = np.split(D, number_of_splits)
            D = np.asarray(D)

        return X, Y, instrument_feature_matrix, tempo, V, D
    else:
        return None, None, None, None, None, None

def import_midi_from_folder(folder, load_from_npy_instead_of_midi, save_imported_midi_as_npy, test_fraction):
    # npyでロードする場合
    if load_from_npy_instead_of_midi:
        if config.LIGHT_MODE:
            print("V: Now Loading ...")
            V_train = np.load(folder+'V_train.npy', allow_pickle=True)
            V_test = np.load(folder+'V_test.npy', allow_pickle=True)

            print("D: Now Loading ...")
            D_train = np.load(folder+'D_train.npy', allow_pickle=True)
            D_test = np.load(folder+'D_test.npy', allow_pickle=True)

            print("T: Now Loading ...")
            T_train = np.load(folder+'T_train.npy', allow_pickle=True)
            T_test = np.load(folder+'T_test.npy', allow_pickle=True)
            
            print("I: Now Loading ...")
            I_train = np.load(folder+'I_train.npy', allow_pickle=True)
            I_test = np.load(folder+'I_test.npy', allow_pickle=True)
            
            # print("Y: Now Loading ...")
            # Y_train = np.load(folder+'Y_train.npy', allow_pickle=True)
            # Y_test = np.load(folder+'Y_test.npy', allow_pickle=True)
            
            print("X: Now Loading ...")
            X_train = np.load(folder+'X_train.npy', allow_pickle=True)
            X_test = np.load(folder+'X_test.npy', allow_pickle=True)
            
            print("c: Now Loading ...")
            c_train = np.load(folder+'c_train.npy', allow_pickle=True)
            c_test = np.load(folder+'c_test.npy', allow_pickle=True)
            
            # print("paths: Now Loading ...")
            # train_paths = np.load(folder+'train_paths.npy', allow_pickle=True)
            # test_paths = np.load(folder+'test_paths.npy', allow_pickle=True)

            return V_train, V_test, D_train, D_test, T_train, T_train, I_train, I_test, None, None, X_train, X_test, c_train, c_test, None, None

        else:
            print("V: Now Loading ...")
            V_train = np.load(folder+'V_train.npy', allow_pickle=True)
            V_test = np.load(folder+'V_test.npy', allow_pickle=True)

            print("D: Now Loading ...")
            D_train = np.load(folder+'D_train.npy', allow_pickle=True)
            D_test = np.load(folder+'D_test.npy', allow_pickle=True)

            print("T: Now Loading ...")
            T_train = np.load(folder+'T_train.npy', allow_pickle=True)
            T_test = np.load(folder+'T_test.npy', allow_pickle=True)
            
            print("I: Now Loading ...")
            I_train = np.load(folder+'I_train.npy', allow_pickle=True)
            I_test = np.load(folder+'I_test.npy', allow_pickle=True)
            
            print("Y: Now Loading ...")
            Y_train = np.load(folder+'Y_train.npy', allow_pickle=True)
            Y_test = np.load(folder+'Y_test.npy', allow_pickle=True)
            
            print("X: Now Loading ...")
            X_train = np.load(folder+'X_train.npy', allow_pickle=True)
            X_test = np.load(folder+'X_test.npy', allow_pickle=True)
            
            print("c: Now Loading ...")
            c_train = np.load(folder+'c_train.npy', allow_pickle=True)
            c_test = np.load(folder+'c_test.npy', allow_pickle=True)
            
            print("paths: Now Loading ...")
            train_paths = np.load(folder+'train_paths.npy', allow_pickle=True)
            test_paths = np.load(folder+'test_paths.npy', allow_pickle=True)

            return V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths

    X_list = []
    Y_list = []
    paths = []
    c_classes = []
    I_list = []
    T_list = []
    V_list = []
    D_list = []
    no_imported = 0

    folder_path = Path(folder)
    class_dirs = list([p for p in folder_path.iterdir() if p.is_dir()])
    class_names = list([p.name for p in folder_path.iterdir() if p.is_dir()])
    if print_anything:print(class_names)

    for i, class_dir in enumerate(class_dirs):
        # print(f"class {class_names[i]}: Now loading ...")
        print("class {}: Now loading ...".format(class_names[i]))
        files = list([p for p in class_dir.glob('**/*.mid')])

        for file in tqdm(files):
            if no_imported >= config.MAX_SONGS:
                break

            if file.suffix == '.mid' or file.suffix == '.midi':
                try:
                    X, Y, I, T, V, D = load_rolls(str(file))
                except mido.midifiles.meta.KeySignatureError as e:
                    print(str(file), e)
                    X, Y, I, T, V, D = None, None, None, None, None, None
                C = i
                        
                if X is not None and Y is not None:
                    X_list.append(X)
                    Y_list.append(Y)
                    I_list.append(I)
                    T_list.append(T)
                    V_list.append(V)
                    D_list.append(D)
                    paths.append(str(file))
                    c_classes.append(C)
                    no_imported += 1

        if no_imported >= config.MAX_SONGS:
            break
    
    if print_anything:print(len(X_list))
    assert(len(X_list) == len(paths))
    assert(len(X_list) == len(c_classes))
    assert(len(X_list) == len(I_list))
    assert(len(X_list) == len(T_list))
    assert(len(X_list) == len(D_list))
    assert(len(X_list) == len(V_list))
        

    unique, counts = np.unique(c_classes, return_counts=True)
    if print_anything:print(dict(zip(unique, counts)))

    V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths = train_test_split(V_list, D_list, T_list, I_list, Y_list, X_list, c_classes, paths, test_size=test_fraction, random_state=42, stratify=c_classes)

    if print_anything:print(c_train)

    # 各スタイルの入力数が同じになるようにする
    equal_mini_songs = True
    if equal_mini_songs:
        splits_per_class = np.zeros((len(class_names),))
        for i, song in enumerate(X_train):
            # クラスのインデックスを取得
            c = c_train[i]
            # 入力数を追加
            splits_per_class[c] += math.ceil(len(song)/(config.OUTPUT_LENGTH//config.MAX_VOICES))

        amount_of_splits = min(splits_per_class)
        # amount_of_splits = int(amount_of_splits * smaller_training_set_factor)
        if print_anything:print(splits_per_class)
        if print_anything:print(amount_of_splits)

        # amount_of_splitsを超えないように各スタイルの曲数を調節して、最終的に使用するデータセットを作成する
        c_train_new = []
        X_train_new = []
        Y_train_new = []
        I_train_new = []
        T_train_new = []
        V_train_new = []
        D_train_new = []
        train_paths_new = []
        splits_per_class_new = np.zeros((len(class_names),))
        for i, song in enumerate(X_train):
            c = c_train[i]
            if splits_per_class_new[c] + math.ceil(len(song)/(config.OUTPUT_LENGTH//config.MAX_VOICES)) <= amount_of_splits:
                c_train_new.append(c_train[i])
                X_train_new.append(X_train[i])
                Y_train_new.append(Y_train[i])
                I_train_new.append(I_train[i])
                T_train_new.append(T_train[i])
                V_train_new.append(V_train[i])
                D_train_new.append(D_train[i])
                train_paths_new.append(train_paths[i])
                splits_per_class_new[c] += math.ceil(len(song)/(config.OUTPUT_LENGTH//config.MAX_VOICES))

        if print_anything:print(splits_per_class_new)
        c_train = c_train_new
        X_train = X_train_new
        Y_train = Y_train_new
        I_train = I_train_new
        T_train = T_train_new
        V_train = V_train_new
        D_train = D_train_new
        train_paths = train_paths_new

    if save_imported_midi_as_npy:
        np.save(folder+'V_train.npy', V_train)
        np.save(folder+'V_test.npy', V_test)

        np.save(folder+'D_train.npy', D_train)
        np.save(folder+'D_test.npy', D_test)

        np.save(folder+'T_train.npy', T_train)
        np.save(folder+'T_test.npy', T_test)
        
        np.save(folder+'I_train.npy', I_train)
        np.save(folder+'I_test.npy', I_test)
        
        np.save(folder+'Y_train.npy', Y_train)
        np.save(folder+'Y_test.npy', Y_test)
        
        np.save(folder+'X_train.npy', X_train)
        np.save(folder+'X_test.npy', X_test)
        
        np.save(folder+'c_train.npy', c_train)
        np.save(folder+'c_test.npy', c_test)
        
        np.save(folder+'train_paths.npy', train_paths)
        np.save(folder+'test_paths.npy', test_paths)

    return V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths

if __name__ == '__main__':
    # _, _, _, _, _, _ = load_rolls('.', '500_miles_high-Chick-Corea_ee.mid')
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = import_midi_from_folder('./data/', False, True, 0.1)   